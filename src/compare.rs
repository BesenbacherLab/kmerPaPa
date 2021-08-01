use std::cmp::Ordering::Equal;
use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::counts::{ExpectedMutationCounts, ObservedMutationCounts};
use crate::io::get_writer;
use crate::observed::Mutation;
use crate::sample::SampledMutations;
use crate::{Float, MutationType};
use crate::counts::DefaultCounter;

pub fn compare_mutations(
    classified_observed_mutations: &[Mutation],
    expected_mutations: &HashMap<String, ExpectedMutationCounts>,
    sampled_mutations: &SampledMutations,
    filter_for_id: Option<&str>,
    alpha : Float,
) -> Result<Vec<ComparedMutations>> {
    let mut result = vec![];
    let observed_mutations =
        tally_up_observed_mutations(classified_observed_mutations, filter_for_id);
    let mut sampled_lof_long : Vec<usize>;
    let mut sampled_lof : DefaultCounter;
    let no_observations = ObservedMutationCounts::default(); // no observed mutations
    for (region, region_expected) in expected_mutations {
        let region_observed = observed_mutations.get(region).unwrap_or(&no_observations);
        let region_sampled = sampled_mutations.get(region).with_context(|| {
            format!("Failed to look up sampled mutations for region {}", &region)
        })?;
        
        sampled_lof_long = vec![];
        let mut observed_lof = 0;
        let mut expected_lof = 0.0;
        for mutation_type in MutationType::iter() {
            if mutation_type == MutationType::Unknown {
                continue; // we don't sample these
            }
            let expected = region_expected.get(mutation_type);
            let observed = region_observed.get(mutation_type);
            let sampled = match region_sampled.get(&mutation_type) {
                Some(v) => v,
                None => {
                    if !expected.eq(&0.0) {
                        eprintln!("[WARNING] mutation_type={} has no sampled values but an expectation value of {} in region {}", mutation_type, expected, region);
                    }
                    continue; // will not add to result
                }
            };
            if mutation_type == MutationType::Nonsense || mutation_type == MutationType::SpliceSite || mutation_type == MutationType::FrameshiftIndel {
                expected_lof += expected;
                observed_lof += observed;
                // sampled is histogram of counts. It's sum should be number of samples.
                // long samples is count for each sample. Has lenght equal to the number of samples. 
                let long_sampled = sampled.to_long();
                if sampled_lof_long.len() == 0 {
                    sampled_lof_long = long_sampled
                } else {
                    for i in 0..sampled_lof_long.len() {
                        sampled_lof_long[i] += long_sampled[i]
                    }
                }
            };

            let p_value = sampled.p_values().n_hits_or_more(observed);
            let expected_lower = sampled.conf_interval(alpha/2.0);
            let expected_upper = sampled.conf_interval(1.0 - alpha/2.0);
            let comparison =
                ComparedMutations::new(region.clone(), mutation_type, observed, expected, expected_lower, expected_upper, p_value);
            result.push(comparison);
        };

        sampled_lof = DefaultCounter::new();
        for x in sampled_lof_long {
            sampled_lof.inc(x);
        }
        //println!("sampled_lof:{:#?}",sampled_lof);
        //println!("{:#?}", attributes);
        let p_value = sampled_lof.p_values().n_hits_or_more(observed_lof);
        let expected_lower = sampled_lof.conf_interval(alpha/2.0);
        let expected_upper = sampled_lof.conf_interval(1.0 - alpha/2.0);
        let comparison = ComparedMutations::new_lof(region.clone(), observed_lof, expected_lof, expected_lower, expected_upper, p_value);
        result.push(comparison);
        
    }
    result.sort_unstable_by(|a, b| a.p_value.partial_cmp(&b.p_value).unwrap_or(Equal));
    Ok(result)
}

pub fn tally_up_observed_mutations(
    mutations: &[Mutation],
    filter_for_id: Option<&str>,
) -> HashMap<String, ObservedMutationCounts> {
    let mut result = HashMap::new();
    for mutation in mutations {
        if let Some(ref region) = mutation.region {
            if let Some(id) = filter_for_id {
                if region != id {
                    continue
                }
            }
            result
                .entry(region.clone())
                .or_insert_with(ObservedMutationCounts::default)
                .add(mutation.mutation_type, 1);
        } // else: No region name, no useful statistics
    }
    result
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComparedMutations {
    region: String,
    mutation_type: &'static str, // I want a string representation in the output file
    observed: usize,
    expected: Float,
    expected_lower : Float,
    expected_upper : Float,
    p_value: Float,
}

const LOF_STRING: &'static str = "LoF";

impl ComparedMutations {
    pub fn new(
        region: String,
        mutation_type: MutationType,
        observed: usize,
        expected: Float,
        expected_lower : Float,
        expected_upper : Float,
        p_value: Float,
    ) -> Self {
        Self {
            region,
            observed,
            expected,
            expected_lower,
            expected_upper,
            p_value,
            mutation_type: mutation_type.as_str(),
        }
    }
    pub fn new_lof(
        region: String,
        observed: usize,
        expected: Float,
        expected_lower : Float,
        expected_upper : Float,
        p_value: Float,
    ) -> Self {
        Self {
            region,
            observed,
            expected,
            expected_lower,
            expected_upper,
            p_value,
            mutation_type: LOF_STRING,
        }
    }
}

pub fn write_to_file(out_path: &str, compared_mutations: &[ComparedMutations]) -> Result<()> {
    let writer = get_writer(out_path)
        .with_context(|| format!("failed to open file {} for writing", out_path))?;
    let mut csv_writer = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_writer(writer);
    for comparison in compared_mutations {
        csv_writer.serialize(comparison)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {}
