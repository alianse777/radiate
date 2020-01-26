#![feature(proc_macro_hygiene, decl_macro)]

#[macro_use] extern crate rocket;
#[macro_use] extern crate rocket_contrib;
extern crate radiate;
extern crate radiate_web;
extern crate serde;
extern crate serde_json;
extern crate serde_derive;

use std::fs::File;
use radiate::prelude::*;
use radiate_web::prelude::*;
use rocket::config::{Config as RConfig, Environment as REnv};
use rocket_contrib::json::{Json, JsonValue};




fn main() {
    // generate_post_data();
    let r_config = RConfig::build(REnv::Staging)
        .address("0.0.0.0")
        .port(42069)
        .finalize()
        .unwrap();

    rocket::custom(r_config)
        .mount("/", routes![run])
        .launch();
}



#[post("/", format = "json", data = "<radiate>")]
fn run(radiate: Json<RadiateDto>) -> Option<JsonValue> {
    
    // unpack the variables
    let env = radiate.0.env?;
    let net = radiate.0.neat?;
    let pop = radiate.0.population?;
    let train = radiate.0.train?;

    // take out the training variables
    let num_evolve = pop.num_evolve?;
    let num_train = train.epochs;
    let learning_rate = train.learning_rate;

    // create a new problem variable
    let xor = XOR::new();

    // set up the population now that it has been recieved
    let (mut solution, _) = Population::<Neat, NeatEnvironment, XOR>::new()
        .constrain(env)
        .populate_clone(net)
        .debug(pop.debug_process?)
        .dynamic_distance(pop.dynamic_distance?)
        .configure(pop.config?)
        .stagnation(pop.stagnation?, pop.genocide?)
        .parental_criteria(pop.parental_criteria?)
        .survivor_criteria(pop.survivor_criteria?)
        .run(|_, fit, num| {
            println!("epoch: {} score: {}", num, fit);
            let diff = 4.0 - fit;
            (diff > 0.0 && diff < 0.01) || num == num_evolve
        }).unwrap();

    // manually train the neural net
    solution.train(&xor.inputs, &xor.answers, learning_rate, Loss::Diff, |iter, _| {
        iter == num_train as usize
    }).unwrap();
    
    // show it
    xor.show(&mut solution);
    Some(json!({"it": "works"}))
}





#[derive(Debug)]
pub struct XOR {
    inputs: Vec<Vec<f32>>,
    answers: Vec<Vec<f32>>
}



impl XOR {
    pub fn new() -> Self {
        XOR {
            inputs: vec![
                vec![0.0, 0.0],
                vec![1.0, 1.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
            ],
            answers: vec![
                vec![0.0],
                vec![0.0],
                vec![1.0],
                vec![1.0],
            ]
        }
    }


    fn show(&self, model: &mut Neat) {
        println!("\n");
        for (i, o) in self.inputs.iter().zip(self.answers.iter()) {
            let guess = model.forward(&i).unwrap();
            println!("Guess: {:.2?} Answer: {:.2}", guess, o[0]);
        }
    }
}



impl Problem<Neat> for XOR {

    fn empty() -> Self { XOR::new() }

    fn solve(&self, model: &mut Neat) -> f32 {
        let mut total = 0.0;
        for (ins, outs) in self.inputs.iter().zip(self.answers.iter()) {
            match model.forward(&ins) {
                Some(guess) => total += (guess[0] - outs[0]).powf(2.0),
                None => panic!("Error in training NEAT")
            }
        }
        4.0 - total
    }

}



#[allow(dead_code)]
fn generate_post_data() {
    
    // create an environment
    let neat_env = NeatEnvironment::new()
        .set_input_size(2)
        .set_output_size(1)
        .set_weight_mutate_rate(0.8)
        .set_edit_weights(0.1)
        .set_weight_perturb(1.75)
        .set_new_node_rate(0.03)
        .set_new_edge_rate(0.04)
        .set_reactivate(0.2)
        .set_activation_functions(vec![
            Activation::LeakyRelu(0.02)
        ]);

    // build the neat network
    let net = Neat::new()
        .input_size(2)
        .batch_size(1)
        .dense_pool(1, Activation::Sigmoid);

    // build the population
    let population = NeatPopulationBuilder::new()
            .num_evolve(100)
            .size(100)
            .dynamic_distance(true)
            .debug_process(true)
            .config(Config {
                inbreed_rate: 0.001,
                crossover_rate: 0.75,
                distance: 0.5,
                species_target: 5
            })
            .stagnation(10)
            .genocide(vec![Genocide::KillWorst(0.9)]);
    
    // put it all together
    let radiate_dto = RadiateDto::new()
            .env(neat_env)
            .train(100, 0.3)        // this has it's own DTO too (TrainDto), but it's small
            .neat(net)
            .population(population)
            .to_json();
    
    // save to a file for testing via Postman
    serde_json::to_writer_pretty(&File::create("post_test.json").unwrap(), &radiate_dto).unwrap();
}