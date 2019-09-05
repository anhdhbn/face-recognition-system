import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="model")
    return graph

GRAPH_PB_PATH = './models/20180402-114759/20180402-114759.pb'
export_path = "./models/export/2"

graph = load_graph(GRAPH_PB_PATH)

image = graph.get_tensor_by_name('model/input:0')
embedding = graph.get_tensor_by_name('model/embeddings:0')
phase_train_placeholder = graph.get_tensor_by_name('model/phase_train:0')


tensor_info_x = tf.saved_model.utils.build_tensor_info(image)
tensor_info_x2 = tf.saved_model.utils.build_tensor_info(phase_train_placeholder)
tensor_info_y = tf.saved_model.utils.build_tensor_info(embedding)

prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={'image': tensor_info_x, 'train': tensor_info_x2},
          outputs={'embedding': tensor_info_y},
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))


with tf.Session(graph=graph) as sess:
    # tf.saved_model.simple_save(
    #     sess,
    #     export_path,
    #     inputs={'images': image, 'phase_train': phase_train_placeholder},
    #     outputs={'embeddings':embedding})
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
        },
        main_op=tf.tables_initializer())
    builder.save()