
extern crate rayon;
extern crate rand;

use std::sync::{Arc};
use rand::Rng;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use super::generation::{Container, Family, Member};
use super::genome::Genome;



//////////////////////////////////////////////////////////////////////////////////////////
/// Note these should not be directly exposed to the user as to avoid confusion with   ///
/// too many knobs to turn to create a population. Instead, provide functions to add   ///
/// them and defaults if they are not added. These are not nessesarily needed options, ///
/// they are add-ons and really only for if you really want to test around with your   ///
/// structure that is evolving, provides users with more options wich is always good   ///
//////////////////////////////////////////////////////////////////////////////////////////



/// Implement a way to pick which way to pick those members who 
/// survice each generation, in other words - pick who gets to stay, 
/// those who do not get to stay 'die off' and are replaced by the children
/// 
/// Fittest - the default option, the top member from each species
/// TopNumber - given a number, keep the top number regardless of species
/// TopPercent - given a percent out of 100, keep the top percent regardless of species
#[derive(Debug, Clone)]
pub enum SurvivalCriteria {
    Fittest,
    TopNumber(usize),
    TopPercent(f32)
}


/// Implement a way to pick parents of children, in other words
/// how is the rest of the population generation after those who 
/// don't survice die out.
/// 
/// BiasedRandom - the default option, statistically pick more fit parents
///                however allow for less fit parents to be picked as well. This is 
///                kinda like putting the members in a species on a curve and randomly 
///                picking from that distribution 
/// OnlySurvivers - those who survive are only allowed to reproduce
/// BestInEachSpecies - only the best in each species are allowed to reproduce
/// MostDifferent - Pick one parent, then find the parent most different from it (structurally) 
///                 and use that as the other parent. Note this could lead to large expansion in population
#[derive(Debug, Clone)]
pub enum PickParents {
    BiasedRandom,
    BestInSpecies,
}



/// Implement the survival enum
impl SurvivalCriteria {


    /// Based on the survival critera, given a vec of containers and families, pick who survives
    #[inline]
    pub fn pick_survivers<T, E>(&self, members: &mut Vec<Container<T, E>>, families: &Vec<Family<T, E>>) -> Option<Vec<Arc<T>>>
        where
            T: Genome<T, E> + Send + Sync + Clone,
            E: Send + Sync
    {
        match self {
            Self::Fittest => {
                Some(families.par_iter()
                    .map(|x| x.lock().unwrap().fittest().1)
                    .collect::<Vec<_>>())
            },
            Self::TopNumber(num) => {
                SurvivalCriteria::get_top_num(*num, members)
            },
            Self::TopPercent(perc) => {
                let num_to_survive = (members.len() as f32 * perc) as usize;
                SurvivalCriteria::get_top_num(num_to_survive, members)
            }
        }
    }



    /// TopNumer and TopPercent are basically the same so this function does the job of both of them,
    /// just convert the percent to a number before calling the function
    #[inline]
    fn get_top_num<T, E>(num_to_keep: usize, members: &mut Vec<Container<T, E>>) -> Option<Vec<Arc<T>>>
        where
            T: Genome<T, E> + Send + Sync + Clone,
            E: Send + Sync
    {
        members.as_mut_slice()
            .par_sort_by(|a, b| {
                b.fitness_score.partial_cmp(&a.fitness_score).unwrap()
            });
        Some((0..num_to_keep)
            .into_iter()
            .map(|i| Arc::clone(&members[i].member))
            .collect())
    }

}




/// implement the pick parents
impl PickParents {


    /// Find two parents to crossover and produce a child
    #[inline]
    pub fn pick_parents<T, E>(&self, inbreed_rate: f32, families: &Vec<Family<T, E>>) -> Option<((f64, Member<T>), (f64, Member<T>))> 
        where
            T: Genome<T, E> + Send + Sync + Clone,
            E: Send + Sync 
    {
        match self {
            Self::BiasedRandom => {
                return Some(self.create_match(inbreed_rate, families))
            },
            Self::BestInSpecies => {
                let mut r = rand::thread_rng();
                let child_one = families.choose(&mut r)?.lock().unwrap().fittest();
                let child_two = families.choose(&mut r)?.lock().unwrap().fittest();
                return Some((child_one, child_two))
            }
        }
    }



    /// pick two parents to breed a child - these use biased random ways of picking 
    /// parents and returns a tuple of tuples where the f64 is the parent's fitness,
    /// and the type is the parent itself
    #[inline]
    fn create_match<T, E>(&self, inbreed_rate: f32, families: &Vec<Family<T, E>>) -> ((f64, Member<T>), (f64, Member<T>)) 
        where
            T: Genome<T, E> + Send + Sync + Clone,
            E: Send + Sync
    {
        let mut r = rand::thread_rng();
        let (species_one, species_two);
        // get two species to pick from taking into account an inbreeding rate - an inbreed can happen without this 
        if r.gen::<f32>() < inbreed_rate {
            let temp = self.get_biased_random_species(&mut r, families).unwrap();
            species_one = Arc::clone(&temp);
            species_two = temp;
        } else {
            species_one = self.get_biased_random_species(&mut r, families).unwrap();
            species_two = self.get_biased_random_species(&mut r, families).unwrap();
        }
        // get two parents from the species, again the parent may be the same 
        let parent_one = self.get_biased_random_member(&mut r, &species_one);
        let parent_two = self.get_biased_random_member(&mut r, &species_two);
        // return the parent tuples
        (parent_one, parent_two)
    }



    /// get a biased random species from the population to get members from
    /// this gets a random species by getting the total adjusted fitness of the 
    /// entire population then finding a random number inside (0, total populatin fitness)
    /// then summing the individual species until they hit that random numer 
    /// Statistically this allows for species with larger adjusted fitnesses to 
    /// have a greater change of being picked for breeding
    #[inline]
    fn get_biased_random_species<T, E>(&self, r: &mut ThreadRng, families: &Vec<Family<T, E>>) -> Option<Family<T, E>> 
        where 
            T: Genome<T, E> + Send + Sync + Clone,
            E: Send + Sync
    {
        // set a result option to none, this will panic! if the result is still none
        // at the end of the function. Then get the total poopulation fitness
        let mut result = None;
        let total = families.iter()
            .fold(0.0, |sum, curr| {
                sum + (*curr).lock().unwrap().get_total_adjusted_fitness()
            });

        // iterate through the species until the iterative sum is at or above the selected
        // random adjusted fitness level
        let mut curr = 0.0;
        let index = r.gen::<f64>() * total;
        for i in families.iter() {
            curr += i.lock().ok()?.get_total_adjusted_fitness();
            if curr >= index {
                result = Some(Arc::clone(i));
                break
            }
        }
        // either return the result, or panic!
        result.or_else(|| Some(Arc::clone(families.first()?)))
    }



    /// Get a biased random member from the species. By summing the fitness scores of the 
    /// members, members with larger fitness scorese are statistically more likely to be picked
    #[inline]
    pub fn get_biased_random_member<T, E>(&self, r: &mut ThreadRng, family: &Family<T, E>) -> (f64, Member<T>)
        where
            T: Genome<T, E> + Send + Sync + Clone,
            E: Send + Sync
    {
        // declare a result which will panic! at the end of the function if there 
        // is no member found, then get the species total fitness score
        let species_lock = family.lock().unwrap();
        let total = species_lock.get_total_adjusted_fitness();
        let index = r.gen::<f64>() * total;
        let (mut result, mut curr) = (None, 0.0);
        // go through each member and see if it's adjusted fitness has pushed it over the edge
        for member in species_lock.members.iter() {
            curr += member.0;
            if curr >= index {
                result = Some(member);
                break
            }
        };
        // either unwrap the result, or if the adjusted fitness of the species was all
        // negative, just take the first member. If the fitness of the species is negative,
        // the algorithm essentially preforms a random search for these biased functions 
        // once the fitness is above 0, it will 'catch on' and start producing biased results
        result.or_else(|| Some(&species_lock.members[0]))
            .and_then(|val| {
                Some((val.0, val.1.clone()
                    .upgrade()
                    .unwrap_or_else(|| panic!("Failed to get random species member."))
                ))
            })
            .unwrap()
    }

}