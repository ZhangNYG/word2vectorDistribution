import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('job_name', '', '')
tf.app.flags.DEFINE_string('ps_hosts', '', '')
tf.app.flags.DEFINE_string('worker_hosts', '','')
tf.app.flags.DEFINE_integer('task_index', 0, '')

ps_hosts = FLAGS.ps_hosts.split(',')
worker_hosts = FLAGS.worker_hosts.split(',')
cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,'worker': worker_hosts})
server = tf.train.Server(
                    {'ps': ps_hosts,'worker': worker_hosts},
                    job_name=FLAGS.job_name,
                    task_index=FLAGS.task_index)

if FLAGS.job_name == 'ps':
  server.join()

with tf.device(tf.train.replica_device_setter(
               worker_device="/job:worker/task:%d" % FLAGS.task_index,
               cluster=cluster_spec)):
  count = tf.Variable(0)
  increment_count = count.assign_add(1)
  init = tf.global_variables_initializer()

sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                            logdir="./checkpoint/",
                            init_op=init,
                            summary_op=None,
                            saver=None,
                            global_step=None,
                            save_model_secs=60)

with sv.managed_session(server.target) as sess:
    sess.run(init)
    step = 1
    while step <= 200000:
        result = sess.run(increment_count)
        if step%10000 == 0:
          print(result)
        if result==2:
          print("!!!!!!!!")
        step += 1
    print("Finished!")

sv.stop() 