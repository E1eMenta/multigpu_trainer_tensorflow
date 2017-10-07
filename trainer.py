import os
import time
import logging
import numpy as np
import tensorflow as tf

CON_GREEN = '\033[92m'
CON_WHITE = "\033[0m"

def create_loger(logfile):
    logger = logging.getLogger('output')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter1 = logging.Formatter('%(asctime)s - %(message)s')
    formatter2 = logging.Formatter('%(message)s')
    fh.setFormatter(formatter1)
    ch.setFormatter(formatter2)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

class Trainer:
    def __init__(self):
        self.use_train_queue = False
        pass

    def create_train_queue(self, queue_input, FIFOsize=256, fixed_shape_list=None, tf_record=False):
        if not(self.use_train_queue):
            self.tf_record = tf_record
            type_list = [var.dtype for var in queue_input]

            if fixed_shape_list == None:
                train_queue = tf.FIFOQueue(FIFOsize, type_list)
            else:
                train_queue = tf.FIFOQueue(FIFOsize, type_list, shapes=fixed_shape_list)

            self.enqueue_op = train_queue.enqueue(queue_input)
            self.q_size = train_queue.size()
            self.FIFOsize = FIFOsize

            dequeue_op = train_queue.dequeue()

            self.use_train_queue = True
            return dequeue_op
        else:
            assert False, "Only one queue is supported"

    def graph_configure(self,
                        train_phs,
                        test_phs,
                        train_model_input,
                        test_model_input,
                        model_fun,
                        get_loss,
                        learning_rate_fun,
                        validator=None,
                        train_report_fun=None,
                        minimization_op=None,
                        optimizer_fun=None
                        ):
        # placeholders to feed train data, can be None if tfrecords are used
        self.train_phs = train_phs
        # placeholders to feed validation data, validation interface
        self.test_phs = test_phs
        # Tensor, which would be set as model input in train phase, need for augmentations and preprocessing
        self.train_model_input = train_model_input
        # Tensor, which would be set as model input in validation phase, need for augmentations and preprocessing
        self.test_model_input = test_model_input
        # Loss function
        self.get_loss = get_loss
        # Model function
        self.model_fun = model_fun
        # Learning rate policy
        self.lr_fun = learning_rate_fun
        # Validation class, must have validate metod, and have auxiliary metod for creation tf nodes
        self.validator = validator
        # Function to print train results, save summary and other every report_steps times
        self.train_report_fun = train_report_fun
        # Train operation, cn clip gradients here, etc
        self.minimization_op = minimization_op
        # Can be set one optimizer for different gpu's, default - adam
        self.optimizer_fun = optimizer_fun


    def train_configure(self,
                        init_data=None,
                        num_gpus=1,
                        num_train_threads=None,
                        batch_size=256,
                        tag="",
                        use_xla=False,
                        gpu_memory_fraction=0.95):

        self.batch_size = batch_size

        if num_train_threads == None:
            self.num_threads = num_gpus
        else:
            self.num_threads = num_train_threads

        self.isTest = tf.placeholder(tf.bool)
        self.batch_idx = tf.Variable(0, trainable=False, name="batch_idx", dtype=tf.int64)
        self.batch_idx_add_ph = tf.placeholder(tf.int64, name="batch_idx_add")
        self.refresh_batch_idx = tf.assign(self.batch_idx, tf.add(self.batch_idx, self.batch_idx_add_ph))

        self.learning_rate = self.lr_fun(self, self.batch_idx)

        vars_on_cpu = True
        last_params = tf.global_variables()
        with tf.device('/cpu:0'):
            if self.optimizer_fun != None:
                self.optimizer = self.optimizer_fun(self, self.learning_rate)
            else:
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-06, beta1=0.90, use_locking=True)
            if vars_on_cpu:
                with tf.variable_scope("", reuse=False):
                    y_train, self.model_name = self.model_fun(self, self.train_model_input, self.isTest)
                    new_params = tf.global_variables()

        with tf.device('/gpu:0'):
            with tf.variable_scope("", reuse=vars_on_cpu):
                y_train, self.model_name = self.model_fun(self, self.train_model_input, self.isTest)
            if not vars_on_cpu:
                new_params = tf.global_variables()

            self.params = [item for item in new_params if item not in last_params]

            loss = self.get_loss(self, self.train_model_input, y_train)
            self.lossT = [loss]
            if self.minimization_op != None:
                self.train_step = [self.minimization_op(self, loss, self.learning_rate)]
            else:
                self.train_step = [self.optimizer.minimize(loss)]

        for idx in range(1, self.num_threads):
            print(idx)
            with tf.device('/gpu:' + str(idx % num_gpus)):
                with tf.variable_scope("", reuse=True):
                    y_train, _ = self.model_fun(self, self.train_model_input, self.isTest)

                loss = self.get_loss(self, self.train_model_input, y_train)

                self.lossT.append(loss)
                with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                    if self.minimization_op != None:
                        self.train_step.append(self.minimization_op(self, loss, self.learning_rate))
                    else:
                        self.train_step.append(self.optimizer.minimize(loss))

        if self.validator != None:
            with tf.variable_scope("", reuse=True):
                self.y_test, self.model_name = self.model_fun(self, self.test_model_input, self.isTest)

            self.validator.configure(self, self.test_model_input, self.y_test)

        self.model_name += tag
        self.log = create_loger(self.model_name + ".log")

        if init_data is not None:
            self.assign_ops = []
            for p in self.params:
                if p.name in init_data:
                    self.log.info(CON_GREEN + "init from pre-trained model: layer " + p.name + CON_WHITE)
                    self.assign_ops += [p.assign(init_data[p.name])]
                else:
                    self.log.info("default init: layer " + p.name)
        else:
            self.assign_ops = None
            for p in self.params:
                self.log.info(p.name)
        glob_init = tf.global_variables_initializer()
        loc_init = tf.local_variables_initializer()
        tf.get_default_graph().finalize()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction, allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
        if use_xla:
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        self.sess = tf.Session(config=config)
        self.sess.run(glob_init)
        self.sess.run(loc_init)
        if self.assign_ops is not None:
            self.sess.run(self.assign_ops)

    def save_weights(self, filename):
        dirName = os.path.dirname(filename)
        try:
            os.makedirs(dirName)
        except:
            pass
        self.log.info(CON_GREEN + "Saving to {} ({} params)".format(filename, len(self.params)) + CON_WHITE)

        weights_dict = {}
        keys = self.params
        sess = self.sess
        weights = sess.run(keys)
        for var, weight in zip(self.params, weights):
            weights_dict[var.name] = weight

        np.savez(filename, **weights_dict)

    def train(self,
              trainDataLoader,
              max_steps=1000000000,
              report_steps=200,
              save_step=None):
        self.log.info("\n\n\n\n\n {} {}".format(self.model_name, self.sess.run(self.learning_rate)))

        finishing = False

        def load_and_enqueue(sess, enqueue_op, coord, dataLoader, tf_record):
            # print(coord.should_stop(), finishing)
            while not coord.should_stop() and not finishing:

                if tf_record:
                    sess.run(enqueue_op)
                    # print("run")
                else:
                    batchData_list = dataLoader.getBatch(self.batch_size)
                    feed_dict = {ph: np.copy(data) for ph, data in zip(self.train_phs, batchData_list)}
                    sess.run(enqueue_op, feed_dict=feed_dict)

        coord = tf.train.Coordinator()

        # print("1", coord.should_stop())
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)
        # while True:
        #     print(self.sess.run(self.sizes))
        #     time.sleep(2.0)
        # print("1.5", coord.should_stop())
        import threading
        if self.use_train_queue:
            # print("2", coord.should_stop())
            # print("queue start")
            prefetchThread = threading.Thread(target=load_and_enqueue,
                                              args=(self.sess,
                                                    self.enqueue_op,
                                                    coord,
                                                    trainDataLoader,
                                                    self.tf_record))
            # print("3", coord.should_stop())
            prefetchThread.isDaemon()
            prefetchThread.start()
            print("Waiting a bit to load the queue...")
            while (self.FIFOsize - 1 > self.sess.run(self.q_size)):
                print("Loaded {} batches".format(self.sess.run(self.q_size)))
                time.sleep(2.0)
            print("Loaded {} batches".format(self.sess.run(self.q_size)))

        try:
            while self.sess.run(self.batch_idx) < max_steps:
                start_time = time.time()

                def train_function(sess, train_op, train_loss, dataLoader):
                    while (train_function.idx < report_steps and not finishing):
                        train_function.idx += 1
                        if self.use_train_queue:
                            loss, _ = sess.run([train_loss, train_op], feed_dict={self.isTest: False})
                        else:
                            batchData_list = dataLoader.getBatch(self.batch_size)
                            feed_dict = {ph: data for ph, data in zip(self.train_phs, batchData_list)}
                            feed_dict[self.isTest] = False
                            loss, _ = sess.run([train_loss, train_op], feed_dict=feed_dict)

                        train_function.losses.append(loss)

                train_function.idx = 0
                train_function.losses = []
                train_threads = []
                for idx in range(0, self.num_threads):
                    train_threads.append(threading.Thread(target=train_function,
                                                          args=([self.sess,
                                                                 self.train_step[idx],
                                                                 self.lossT[idx],
                                                                 trainDataLoader])))

                for t in train_threads:
                    t.start()
                for t in train_threads:
                    t.join()

                self.sess.run(self.refresh_batch_idx, feed_dict={self.batch_idx_add_ph: train_function.idx})

                train_done = time.time()
                gpu_time = train_done - start_time
                if self.use_train_queue:
                    q_size_before_test = self.sess.run(self.q_size)
                if self.train_report_fun != None:
                    self.train_report_fun(self, train_function.losses, self.batch_idx, self.log)
                if save_step != None:
                    idx = self.sess.run(self.batch_idx)
                    if idx % save_step < report_steps:
                        self.save_weights("saved/weights_" + self.model_name + "_{}.npz".format(idx))
                self.log.info("time for batch: {0:.4f} ms".format(1000.0 * gpu_time / float(report_steps)))

                if self.validator != None:
                    self.validator.validate(self, self.test_phs, self.y_test, self.sess, self.log)
                if self.use_train_queue:
                    self.log.info("Queue num: {} -> {} batches".format(q_size_before_test, self.sess.run(self.q_size)))

        except KeyboardInterrupt:
            finishing = True
            self.save_weights("weights_" + self.model_name + ".nnz")
            import signal
            os.kill(os.getpid(), signal.SIGKILL)

            for t in train_threads:
                t.join()
            coord.request_stop()
            coord.join([prefetchThread])

            # exit(0)

        coord.request_stop()
        coord.join([t1])
        return