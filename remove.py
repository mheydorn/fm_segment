#!/usr/bin/env python3

import argparse
import os
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.core.framework import graph_pb2

def remove(nodes, node_type):
    first_node_type_previous_layer = ""
    first_node_type = -1
    found_first_node_type = False
    last_node_type_input = -1
    last_layer_using_node_type = -1
    last_node_type = -1
    for i, node in enumerate(nodes):
        if node_type in node.name:
            last_node_type = i
            if not found_first_node_type:
                first_node_type = i
                found_first_node_type = True
        else:
            if not found_first_node_type:
                first_node_type_previous_layer = node.name
        try:
            for j, n in enumerate(node.input):
                if node_type in n:
                    last_node_type_input = j
                    last_layer_using_node_type = i
        except UnicodeEncodeError:
            print("ERROR ")
    if(found_first_node_type and 
       last_layer_using_node_type != -1 and 
       first_node_type != -1 and 
       last_node_type_input != -1 and 
       first_node_type_previous_layer != "" and 
       last_node_type != -1):
        nodes[last_layer_using_node_type].input[last_node_type_input] = first_node_type_previous_layer
        first_half = nodes[:first_node_type]
        second_half = nodes[last_node_type+1:]
        return first_half + second_half, True # True means we removed node_type layers
    else:
        return nodes, False # False means we didn't remove any node_type layers

def freeze_removed(model, output, node_types):
    graph = tf.GraphDef()
    with tf.gfile.Open(model, 'rb') as f:
        data = f.read()
        graph.ParseFromString(data)
    nodes = graph.node

    changed = False
    for node_type in node_types:
        nodes, node_changed = remove(nodes, node_type)
        changed = changed or node_changed

    if changed:
        output_graph = graph_pb2.GraphDef()
        output_graph.node.extend(nodes)
        with tf.gfile.GFile(output, 'w') as f:
            f.write(output_graph.SerializeToString())
    else:
        #shutil.copyfile(model, output)
        print("There are no", node_type, "layers to remove. File was copied.")
    return changed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",
                        dest='model',
                        type=str, 
                        default="", 
                        help="Input model file to parse")
    parser.add_argument("-o", "--output", 
                        dest='output',
                        type=str, 
                        default="model_without_dropout.pb", 
                        help="Output model file")
    parser.add_argument("-n", "--node-type", 
                        dest='node_type',
                        action='append',
                        default=[], 
                        help="Node type of layers to remove from model")
    args = parser.parse_args()
    if not args.model:
        print("You must provide a model to be parsed")
        os._exit(1)
    if not args.node_type:
        args.node_type = ['dropout', 'sub']
    model = os.path.abspath(args.model)
    output = os.path.abspath(args.output)
    freeze_removed(model, output, args.node_type)

