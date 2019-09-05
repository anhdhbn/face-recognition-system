import tensorflow as tf
from numpy import dot
from numpy.linalg import norm
import glob
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
import tensorflow.contrib.tensorrt as trt
import time

# GRAPH_PB_PATH = './models/tensorrt.pb'
GRAPH_PB_PATH = './models/20180402-114759/20180402-114759.pb'
# GRAPH_PB_PATH = './models/tensorrt32/saved_model.pb'
# sess = tf.InteractiveSession()

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

graph = load_graph(GRAPH_PB_PATH)
outputs = ["model/embeddings"]

# print([n.name for n in graph.as_graph_def().node])


phase_train_placeholder = graph.get_tensor_by_name('model/phase_train:0')
image = graph.get_tensor_by_name('model/input:0')
embedding = graph.get_tensor_by_name('model/embeddings:0')

with tf.Session(graph=graph) as sess:
    frozen_graph = tf.graph_util.convert_variables_to_constants(sess,
            graph.as_graph_def(), output_node_names=outputs)

    with tf.gfile.FastGFile("./models/frozen_model.pb", "wb") as f:
        f.write(frozen_graph.SerializeToString())
# trt.create_inference_graph(GRAPH_PB_PATH)

    trt_graph = trt.create_inference_graph(input_graph_def=frozen_graph,
        outputs=outputs,
        max_batch_size=256,
        max_workspace_size_bytes=4*(10**9),
        precision_mode="FP32")

    with tf.gfile.FastGFile("./models/tensorrt2.pb", "wb") as f:
        f.write(frozen_graph.SerializeToString())

    all_nodes = len([1 for n in frozen_graph.node])
    print("numb. of all_nodes in frozen graph:", all_nodes)

# # check how many ops that is converted to TensorRT engine
# trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
# print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
# all_nodes = len([1 for n in trt_graph.node])
# print("numb. of all_nodes in TensorRT graph:", all_nodes)
# # print(phase_train_placeholder)

def preprocess_image(image_path):
    ''' Load the image, resize and normalized it
    Note: The way we normalize the image here should be
    consistant with the way we nomalize the images while training
    '''

    img = load_img(image_path, target_size=(160, 160))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0).astype(float)
    img = (img - 127.5) / 128.0
    return img

def get_embedding(image_path):
    ''' Helper function to get an embedding of a image
    Load and pre-process the image
    Feed the image into the network and get the embedding vector
    '''

    img = preprocess_image(image_path)

    with tf.Session(graph=graph) as sess:
        emb = sess.run(embedding, feed_dict={image: img, phase_train_placeholder: False})
        return emb.squeeze()

def findDistance(cand, test):
    ''' Compute cosin between 2 vectors
    '''
    a = cand
    b = test

    cos_sin = dot(a, b)/(norm(a)*norm(b))
    return np.arccos(cos_sin)
#     euclidean_distance = cand - test
#     euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
#     euclidean_distance = np.sqrt(euclidean_distance)
#     return cosine_similarity(cand, test)


# print(help(tf.saved_model.signature_constants))

# ,method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME

# prediction_signature = (
#     tf.saved_model.signature_def_utils.build_signature_def(
#         inputs={'images': tensor_info_x},
#         outputs={'embeddings': tensor_info_y}))

# with tf.Session(graph=graph) as sess:
#     export_path = "./models/export"
#     # builder = tf.saved_model.builder.SavedModelBuilder(export_path)
#     # builder.add_meta_graph_and_variables(
#     #     sess, [tf.saved_model.tag_constants.SERVING],
#     #     signature_def_map={
#     #         'predict':
#     #             prediction_signature,
#     #     },
#     #     main_op=tf.tables_initializer())
#     # builder.save()
#     tf.saved_model.simple_save(
#         sess,
#         export_path,
#         inputs={'images': image},
#         outputs={'embeddings':embedding})



# test_img = './test/test_img.png'
# db_imgs = glob.glob('test/db/*.png')

# start  = time.time()
# test_eb = get_embedding(test_img)
# db_eb = [get_embedding(img_path) for img_path in db_imgs]

# end = time.time()

# dt = [findDistance(test_eb, eb) for eb in db_eb]
# min_idx = np.argmin(dt)
# dst_img = load_img(db_imgs[min_idx], target_size=(160, 160))

# img = load_img(test_img, target_size=(160, 160))

# print(dt[min_idx])



# print(end - start)