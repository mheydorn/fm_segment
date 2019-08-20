#!/usr/bin/env python3

import os
import argparse
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import numpy as np
import remove

dir = os.path.dirname(os.path.realpath(__file__))

def create_pb(modelName):
    pb_name = modelName +".pb"
    freeze_graph('./', 
                        'Softmax',
                        modelName,
                        pb_name)
    model = os.path.abspath(pb_name)
    output = os.path.abspath(pb_name)
    remove.freeze_removed(model, output, ['dropout', 'sub'])
    


def convert_placeholders_to_constants(input_graph_def,
                                      placeholder_to_value_map):
  """Replaces placeholders in the given tf.GraphDef with constant values.

  Args:
    input_graph_def: GraphDef object holding the network.
    placeholder_to_value_map: A map from the names of placeholder tensors in
      `input_graph_def` to constant values.

  Returns:
    GraphDef containing a simplified version of the original.
  """

  output_graph_def = tf.GraphDef()

  for node in input_graph_def.node:
    output_node = tf.NodeDef()
    if node.op == "Placeholder" and node.name in placeholder_to_value_map:
      print("Found a placeholder that matches!", node.name)
      output_node.op = "Const"
      output_node.name = node.name
      dtype = node.attr["dtype"].type
      data = np.asarray(placeholder_to_value_map[node.name],
                        dtype=tf.as_dtype(dtype).as_numpy_dtype)
      output_node.attr["dtype"].type = dtype
      output_node.attr["value"].CopyFrom(tf.AttrValue(
          tensor=tf.contrib.util.make_tensor_proto(data,
                                                   dtype=dtype,
                                                   shape=data.shape)))
    else:
      output_node.CopyFrom(node)

    output_graph_def.node.extend([output_node])

  return output_graph_def

def freeze_graph(model_dir, output_node_names, input_checkpoint, model_name=None):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1
    if not input_checkpoint:
        print("You need to supply the name of the checkpoint file to --input_checkpoint.")
        return -2
    if not model_name:
        model_name = "model.pb"
        print("No model name supplied. Using 'model.pb' as the model name.")

    # We retrieve our checkpoint fullpath
    #checkpoint = tf.train.get_checkpoint_state(model_dir)
    #input_checkpoint = checkpoint.model_checkpoint_path
    #input_checkpoint = "/data/laitram/hoso_clump/export/model/model_with_test_accuracy_0.9621112_.ckpt"
    input_checkpoint = os.path.abspath(input_checkpoint)
    
    # We precise the file fullname of our freezed graph
    #absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    absolute_model_dir = os.path.abspath(model_dir)
    #output_graph = absolute_model_dir + "/" + model_name
    output_graph = absolute_model_dir + "/" + model_name

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    sess = tf1.InteractiveSession(graph=tf.Graph())
    #with tf.Session(graph=tf.Graph()) as sess:
    with sess.as_default():
      with sess.graph.as_default():
        # We import the meta graph in the current default Graph
        saver = tf1.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf1.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf1.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 
        #output_graph_def = tf.get_default_graph().as_graph_def()
        
        #output_graph_def = convert_placeholders_to_constants(output_graph_def,{"keep_probabilty": 1 })
        output_graph_def = convert_placeholders_to_constants(output_graph_def,{"keep_prob": 1 })

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        #print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--model_dir", type=str, default="", help="Model folder to export")
    parser.add_argument("-o", "--output_node_names", type=str, default="", help="The name of the output nodes, comma separated.")
    parser.add_argument("-i", "--input_checkpoint", type=str, default="", help="The name of the checkpoint file to convert to a pb model")
    parser.add_argument("-m", "--model_name", type=str, default="", help="The name of the model to save (ie. fm_model.pb)")
    args = parser.parse_args()

    freeze_graph(args.model_dir, 
                 args.output_node_names,
                 args.input_checkpoint,
                 args.model_name)
