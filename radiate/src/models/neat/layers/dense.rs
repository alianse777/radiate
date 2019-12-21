extern crate rand;
extern crate uuid;

use std::collections::HashSet;
use std::mem;
use std::fmt;
use std::any::Any;
use std::sync::{Arc, RwLock};
use rand::Rng;
use rand::seq::SliceRandom;
use uuid::Uuid;
use super::{
    layertype::LayerType,
    layer::Layer,
};
use super::super::{
    neuron::Neuron,
    neatenv::NeatEnvironment,
    neurontype::NeuronType,
    activation::Activation
};

use crate::Genome;



pub struct DenseIter<'a> {
    stack: Vec<&'a *mut Neuron>,
    seen: HashSet<&'a *mut Neuron>
}

impl<'a> Iterator for DenseIter<'a> {
    type Item = &'a *mut Neuron;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            let curr_ptr = self.stack.pop()?;
            for key in (**curr_ptr).outgoing.keys() {
                if !self.seen.contains(key) {
                    self.stack.push(key);
                    self.seen.insert(key);
                }
            }
            Some(curr_ptr)
        }
    }
}




#[derive(Debug)]
pub struct Dense {
    pub inputs: Vec<*mut Neuron>,
    pub outputs: Vec<*mut Neuron>,
    pub size: i32,
    pub layer_type: LayerType,
    pub activation: Activation
}



impl Dense {

    
    /// create a new fully connected dense layer.
    /// Each input is connected to each output with a randomly generated weight attached to the connection
    #[inline]
    pub fn new(num_in: i32, num_out: i32, layer_type: LayerType, activation: Activation) -> Self {
        let layer = Dense {
            inputs: (0..num_in)
                .into_iter()
                .map(|_| Neuron::new(Uuid::new_v4(), NeuronType::Input, Activation::Sigmoid).as_mut_ptr())
                .collect(),
            outputs: (0..num_out)
                .into_iter()
                .map(|_| Neuron::new(Uuid::new_v4(), NeuronType::Output, Activation::Sigmoid).as_mut_ptr())
                .collect(),
            size: num_in + num_out,
            layer_type,
            activation
        };
        
        let mut r = rand::thread_rng();
        for input_ptr in layer.inputs.iter() {
            for output_ptr in layer.outputs.iter() {
                layer.connect(input_ptr, output_ptr, r.gen::<f64>());
            }
        }

        layer
    }



    /// Connect two raw pointers to each other in the graph
    #[inline]
    pub fn connect(&self, src_ptr: &*mut Neuron, dst_ptr: &*mut Neuron, weight: f64) {
        unsafe {
            (**src_ptr).outgoing.insert(*dst_ptr, weight);
            (**dst_ptr).incoming.insert(*src_ptr, None);
        }
    }



    pub fn iter(&self) -> DenseIter {
        let mut stack = Vec::new();
        let mut seen = HashSet::new();
        for input_ptr in self.inputs.iter() {
            stack.push(input_ptr);
            seen.insert(input_ptr);
        }

        DenseIter { stack, seen }       
    }


    pub fn contains(&self, other: &*mut Neuron) -> bool {
        self.iter().any(|x| x == other)
    }


    pub fn get(&self, ptr: &*mut Neuron) -> Option<&*mut Neuron> {
        for neuron_ptr in self.iter() {
            if neuron_ptr == ptr {
                return Some(neuron_ptr);
            }
        }
        None
    }

    

    /// reset all the neurons in the network so they can be fed forward again
    #[inline]
    unsafe fn reset_neurons(&self) {
        for neuron_ptr in self.iter() {
            (**neuron_ptr).reset_neuron();
        }
    }   
    


    /// Add a node to the network by getting a random edge 
    /// and inserting the new node inbetween that edge's source
    /// and destination nodes. The old weight is pushed forward 
    /// while the new weight is randomly chosen and put between the 
    /// old source node and the new node
    #[inline]
    pub fn add_node(&mut self, activation: Activation) {
        unsafe {
            // create a new node to insert inbetween the sending and receiving nodes 
            let new_node = Neuron::new(Uuid::new_v4(), NeuronType::Hidden, activation).as_mut_ptr();
            let src_ptr = loop {
                let temp = self.random_neuron();
                if (**temp).neuron_type != NeuronType::Output {
                    break temp;
                }
            };
            let dst_ptr = (**src_ptr).random_connection();
            let conn_weight = *(**src_ptr).outgoing.get(dst_ptr).unwrap();

            self.connect(src_ptr, &new_node, 1.0);
            self.connect(&new_node, dst_ptr, conn_weight);

            (**src_ptr).inactive.insert(*dst_ptr, conn_weight);
            (**src_ptr).outgoing.remove(dst_ptr);
            (**dst_ptr).incoming.remove(src_ptr);  
        }
    }



    /// add a connection to the network. Randomly get a sending node that cannot 
    /// be an output and a receiving node which is not an input node, the validate
    /// that the desired connection can be made. If it can be, make the connection
    /// with a weight of .5 in order to minimally impact the network 
    #[inline]
    pub fn add_edge(&mut self) {
        unsafe {
            // get a valid sending neuron
            let src_ptr = loop {
                let temp = self.random_neuron();
                if (**temp).neuron_type != NeuronType::Output {
                    break temp;
                }
            };
            let dst_ptr = loop {
                let temp = self.random_neuron();
                if (**temp).neuron_type != NeuronType::Input {
                    break temp;
                }
            };

            // determine if the connection to be made is valid 
            // if the connection is valid, make it and wire the nodes to each
            if self.valid_connection(src_ptr, dst_ptr) {
                self.connect(src_ptr, dst_ptr, rand::thread_rng().gen::<f64>());
            }
        }
    }



    /// Test whether the desired connection is valid or not by testing to see if 
    /// 1.) it is recursive
    /// 2.) the connection already exists
    /// 3.) the desired connection would create a cycle in the graph
    /// if these are all false, then the connection can be made
    #[inline]
    unsafe fn valid_connection(&self, sending: &*mut Neuron, receiving: &*mut Neuron) -> bool {
        if sending == receiving {
            return false
        } else if self.exists(sending, receiving) {
            return false
        } else if self.cyclical(sending, receiving) {
            return false
        }
        true
    }



    /// check to see if the connection to be made would create a cycle in the graph
    /// and therefore make it network invalid and unable to feed forward
    #[inline]
    unsafe fn cyclical(&self, sending: &*mut Neuron, receiving: &*mut Neuron) -> bool {
        // dfs stack which gets the receiving Neuron<dyn neurons> outgoing connections
        let mut stack = (**receiving).outgoing
            .keys()
            .map(|x| x)
            .collect::<Vec<_>>();
       
            // while the stack still has nodes, continue
        while stack.len() > 0 {
            
            // if the current node is the same as the sending, this would cause a cycle
            // else add all the current node's outputs to the stack to search through
            let curr = stack.pop().unwrap();
            if curr == sending {
                return true;
            }
            for i in (**curr).outgoing.keys() {
                stack.push(i);
            }
        }
        false
    }



    /// check if the desired connection already exists within he network, if it does then
    /// we should not be creating the connection.
    #[inline]
    unsafe fn exists(&self, sending: &*mut Neuron, receiving: &*mut Neuron) -> bool {
        (**sending).outgoing.keys().any(|x| x == receiving) || (**receiving).outgoing.keys().any(|x| x == sending)
    }



    /// get a random node from the network - the hashmap does not have a idomatic
    /// way to do this so this is a workaround. Returns the innovation number of the node
    /// in order to satisfy rust borrow rules
    #[inline]
    fn random_neuron(&self) -> &*mut Neuron {
        let index = rand::thread_rng().gen_range(0, self.size) as usize;
        for (i, neuron_ptr) in self.iter().enumerate() {
            if i == index {
                return neuron_ptr;
            }
        }
        panic!("Random neuron failed.");
    }



    /// give input data to the input nodes in the network and return a vec
    /// that holds the innovation numbers of the input nodes for a dfs traversal 
    /// to feed forward those inputs through the network
    #[inline]
    unsafe fn give_inputs(&self, data: &Vec<f64>) -> Vec<*mut Neuron> {
        assert!(data.len() == self.inputs.len());
        let mut path = Vec::with_capacity(self.inputs.len());
        for (input_ptr, val) in self.inputs.iter().zip(data.iter()) {
            (**input_ptr).value = Some(*val);
            path.push(*input_ptr);
        }
        path
    }



    /// Edit the weights in the network randomly by either uniformly perturbing
    /// them, or giving them an entire new weight all together
    #[inline]
    fn edit_weights(&mut self, editable: f32, size: f64) {
        let mut r = rand::thread_rng();
        unsafe {
            for neuron_ptr in self.iter() {
                for (outgoing_ptr, weight) in (**neuron_ptr).outgoing.iter_mut() {
                    if r.gen::<f32>() < editable {
                        *weight = r.gen::<f64>();
                    } else {
                        *weight *= r.gen_range(-size, size);
                    }
                }
            }
        }
    }



    pub fn see(&self) {
        unsafe { 
            for neuron_ptr in self.iter() {
                println!("{:?}", (**neuron_ptr));
            }
        }
    }




}




impl Layer for Dense {
    /// Feed a vec of inputs through the network, will panic! if 
    /// the shapes of the values do not match or if something goes 
    /// wrong within the feed forward process.
    #[inline]
    fn propagate(&mut self, data: &Vec<f64>) -> Option<Vec<f64>> {
        unsafe {
            // reset the network by clearing the previous outputs from the neurons 
            // this could be done more efficently if i didn't want to implement backprop
            // or recurent nodes, however this must be done this way in order to allow for the 
            // needed values for those algorithms to remain while they are needed 
            // give the input data to the input neurons and return back 
            // a stack to do a graph traversal to feed the inputs through the network
            self.reset_neurons();
            let mut path = self.give_inputs(data);

            // while the path is still full, continue feeding forward 
            // the data in the network, this is basically a dfs traversal
            while path.len() > 0 {
            
                // remove the top elemet to propagate it's value
                let curr_node = &path.pop()?;
            
                // no node should be in the path if it's value has not been set 
                // iterate through the current nodes outgoing connections 
                // to get its value and give that value to it's connected node
                if let Some(val) = (**curr_node).value {
                    for (neuron_ptr, weight) in (**curr_node).outgoing.iter() {
            
                        // if the currnet edge is active in the network, we can propagate through it
                        (**neuron_ptr).incoming.insert(*curr_node, Some(val * weight));
        
                        // if the node can be activated, activate it and store it's value
                        // only activated nodes can be added to the path, so if it's activated
                        // add it to the path so the values can be propagated through the network
                        if (**neuron_ptr).is_ready() {
                            (**neuron_ptr).activate();
                            path.push(*neuron_ptr);
                        }
                    }
                }
            }
            
            // once we've made it through the network, the outputs should all
            // have calculated their values. Gather the values and return the vec
            let mut network_output = Vec::with_capacity(self.outputs.len());
            for neuron_ptr in self.outputs.iter() {
                let node_val = (**neuron_ptr).value?; 
                network_output.push(node_val);
            }
            Some(network_output)
        }
    }


    /// Backpropagation algorithm, transfer the error through the network and change the weights of the
    /// edges accordinly, this is pretty straight forward due to the design of the neat graph
    #[inline]
    fn backprop(&mut self, error: &Vec<f64>, learning_rate: f64) -> Option<Vec<f64>> {
        // feed forward the input data to get the output in order to compute the error of the network
        // create a dfs stack to step backwards through the network and compute the error of each neuron
        // then insert that error in a hashmap to keep track of innov of the neuron and it's error 
        // 
        unsafe  {
            let mut path = Vec::new();
            for (i, neuron_ptr) in self.outputs.iter().enumerate() {
                (**neuron_ptr).error = Some(error[i]);
                path.push(*neuron_ptr);
            }

            // step through the network backwards and adjust the weights
            while path.len() > 0 {
              
                // get the current node and it's error 
                let curr_node = &path.pop()?;
                let curr_node_error = (**curr_node).error? * learning_rate;
              
                // iterate through each of the incoming edes to this neuron and adjust it's weight
                // and add it's error to the errros map
                for (incoming_ptr, val)  in (**curr_node).incoming.iter() {
              
                    // if the current edge is active, then it is contributing to the error and we need to adjust it
                    let step = curr_node_error * (**curr_node).deactivate();
                    
                    let mut incoming_weight = *(**incoming_ptr).outgoing.get(curr_node)?;
                    // add the weight step (gradient) * the currnet value to the weight to adjust the weight by the error
                    incoming_weight += step * (*val)?;
                    (**incoming_ptr).error = Some(incoming_weight * curr_node_error);
                    (**incoming_ptr).outgoing.insert(*curr_node, incoming_weight);
                    path.push(*incoming_ptr);
                }
            }
            let mut output = Vec::with_capacity(self.inputs.len());
            for input_ptr in self.inputs.iter() {
                output.push((**input_ptr).error?);
            }
            Some(output)
        }
    }


    
    fn as_ref_any(&self) -> &dyn Any
        where Self: Sized + 'static
    {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn Any
        where Self: Sized + 'static
    {
        self
    }

    fn shape(&self) -> (usize, usize) {
        (self.inputs.len(), self.outputs.len())
    }

}


impl Genome<Dense, NeatEnvironment> for Dense
    where Dense: Layer
{
    fn crossover(child: &Dense, parent_two: &Dense, env: &Arc<RwLock<NeatEnvironment>>, crossover_rate: f32) -> Option<Dense> {
        let mut new_child = child.clone();
        unsafe {
            let set = (*env).read().ok()?;
            let mut r = rand::thread_rng();
            if r.gen::<f32>() < crossover_rate {

                for neuron_ptr in new_child.iter() {
                    if parent_two.contains(neuron_ptr) {
                        let parent_neuron_ptr = parent_two.get(neuron_ptr)?;
                        for (outgoing_ptr, weight) in (**neuron_ptr).outgoing.iter_mut() {
                            if (**parent_neuron_ptr).outgoing.contains_key(outgoing_ptr) && r.gen::<f32>() < 0.5 {
                                let weight = (**parent_neuron_ptr).outgoing.get(outgoing_ptr)?;
                                (**neuron_ptr).outgoing.insert(*outgoing_ptr, *weight);
                            }
                        }

                        for (inactive_ptr, weight) in (**neuron_ptr).inactive.iter_mut() {
                            if (**parent_neuron_ptr).inactive.contains_key(inactive_ptr) && r.gen::<f32>() < set.reactivate? {
                                (**neuron_ptr).outgoing.insert(*inactive_ptr, *weight);
                                (**inactive_ptr).incoming.insert(*neuron_ptr, None);
                                (**neuron_ptr).inactive.remove(inactive_ptr);
                            }
                        }
                    }
                }
            } else {
                
                // if a random number is less than the edit_weights parameter, then edit the weights of the network edges
                // add a possible new node to the network randomly 
                // attempt to add a new edge to the network, there is a chance this operation will add no edge
                if r.gen::<f32>() < set.weight_mutate_rate? {
                    new_child.edit_weights(set.edit_weights?, set.weight_perturb?);
                }
                
                // if the layer is a dense pool then it can add nodes and connections to the layer as well
                if new_child.layer_type == LayerType::DensePool {
                    if r.gen::<f32>() < set.new_node_rate? {
                        let act_func = *set.activation_functions.choose(&mut r)?;
                        new_child.add_node(act_func);
                    }
                    if r.gen::<f32>() < set.new_edge_rate? {
                        new_child.add_edge();
                    }
                }
            }
        }
        Some(new_child)
    }



    fn distance(one: &Dense, two: &Dense, _: &Arc<RwLock<NeatEnvironment>>) -> f64 {
        let mut similar = 0.0;
        unsafe {
            for neuron_ptr in one.iter() {
                let other_ptr = two.get(neuron_ptr);
                match other_ptr {
                    Some(ptr) => {
                        for (key_ptr, weight) in (**neuron_ptr).outgoing.iter() {
                            if (**ptr).outgoing.contains_key(key_ptr) {
                                similar += 1.0;
                            }
                        }
                    },
                    _ => similar += 0.0
                }
            }
        }

        let one_score = similar / one.size as f64;
        let two_score = similar / two.size as f64;
        2.0 - (one_score + two_score)
    }
}




/// Implement clone for the neat neural network in order to facilitate 
/// proper crossover and mutation for the network
impl Clone for Dense {
    fn clone(&self) -> Self {
        Dense {
            inputs: self.inputs
                .iter()
                .map(|x| {
                    unsafe { (**x).clone().as_mut_ptr() }
                })
                .collect(),
            outputs: self.outputs
                .iter()
                .map(|x| {
                    unsafe { (**x).clone().as_mut_ptr() }
                })
                .collect(),
            size: self.size,
            layer_type: self.layer_type.clone(),
            activation: self.activation.clone()
        }
    }
}
/// Because the tree is made out of raw mutable pointers, if those pointers
/// are not dropped, there is a severe memory leak, like possibly gigs of
/// ram over only a few generations depending on the size of the generation
/// This drop implementation will recursivley drop all nodes in the tree 
impl Drop for Dense {
    fn drop(&mut self) { 
        unsafe {
            for node in self.iter() {
                drop(Box::from_raw(*node));
            }
        }
    }
}
/// These must be implemneted for the network or any type to be 
/// used within seperate threads. Because implementing the functions 
/// themselves is dangerious and unsafe and i'm not smart enough 
/// to do that from scratch, these "implmenetaions" will get rid 
/// of the error and realistically they don't need to be implemneted for the
/// program to work
unsafe impl Send for Dense {}
unsafe impl Sync for Dense {}
/// Implement partialeq for neat because if neat itself is to be used as a problem,
/// it must be able to compare one to another
impl PartialEq for Dense {
    fn eq(&self, other: &Self) -> bool {
        if self.size != other.size {
            return false;
        }
        for (one_ptr, two_ptr) in self.iter().zip(other.iter()) {
            if one_ptr != two_ptr {
                return false;
            }
        }
        true
    }
}
/// Simple override of display for neat to debug a little cleaner 
impl fmt::Display for Dense {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unsafe {
            let address: u64 = mem::transmute(self);
            write!(f, "Dense=[{}, {}]", address, self.size)
        }
    }
}


