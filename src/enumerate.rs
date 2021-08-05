use std::collections::hash_map::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Write};

use anyhow::{Context, Result};

use mutexpect::{possible_mutations, MutationEvent, SeqAnnotation};
use pattern_partition_prediction::{PaPaPred, PaPaPredIndel};
use twobit::TwoBitFile;

use crate::error::ParseError;
use crate::io::{get_reader, get_writer};
use crate::{Float, MutationType};

type PossibleMutations = HashMap<String, Vec<MutationEvent>>;

pub fn enumerate_possible_mutations(
    annotations: &[SeqAnnotation],
    ref_genome: &TwoBitFile,
    mutation_rates: &PaPaPred,
    indel_mutation_rates: &Option<PaPaPredIndel>,
    scaling_factor: f32,
    drop_nan: bool,
    filter_for_id: Option<&str>,
    include_intronic : bool,
    include_unknown : bool,
    filter_plof : bool,
) -> Result<PossibleMutations> {
    let mut result = HashMap::new();
    let radius = mutation_rates.radius();
    for annotation in annotations {
        if let Some(id) = filter_for_id {
            if annotation.name != id {
                continue;
            }
        }
        let start = annotation.range.start - radius;
        let stop = annotation.range.stop + radius + 1;
        let seq = ref_genome.sequence(&annotation.chr, start, stop)?;
        match possible_mutations(&seq, &annotation, mutation_rates, indel_mutation_rates, drop_nan, include_intronic, include_unknown, filter_plof) {
            Ok(mut mutations) => {
                if scaling_factor != 1.0 {
                    for mutation in &mut mutations {
                        mutation.probability = 1.0 - (1.0 - mutation.probability).powf(scaling_factor);
                        assert!(mutation.probability>=0.0);
                        assert!(mutation.probability<=1.0);
                        //mutation.probability *= scaling_factor;
                        //mutation.probability = f32::min(mutation.probability, 1.0)
                    }
                }
                result.insert(annotation.name.clone(), mutations);
            }
            Err(e) => {
                eprintln!(
                    "[WARNING] Skipping faulty annotation {}: {}",
                    annotation.name, e
                );
            }
        }
    }

    Ok(result)
}

pub fn write_to_file(out_path: &str, possible_mutations: &PossibleMutations) -> Result<()> {
    let writer = get_writer(out_path)
        .with_context(|| format!("failed to open file {} for writing", out_path))?;
    let mut buf_writer = BufWriter::new(writer);
    for (name, mutations) in possible_mutations {
        buf_writer.write_all(format!("#{}\n", name).as_bytes())?;
        for mutation in mutations {
            buf_writer.write_all(
                format!(
                    "{}:{}\n",
                    mutation.mutation_type as usize, mutation.probability
                )
                .as_bytes(),
            )?;
        }
    }

    Ok(())
}

pub fn read_from_file(in_path: &str) -> Result<PossibleMutations> {
    let mut result: PossibleMutations = HashMap::new();
    let reader = get_reader(in_path)
        .with_context(|| format!("failed to open file {} for reading", in_path))?;
    let bufreader = BufReader::new(reader);
    let mut current_gene: Option<String> = None;
    for (line_no, line) in bufreader.lines().enumerate() {
        if let Ok(line) = line {
            if line.starts_with('#') {
                current_gene = Some(line[1..].to_string());
                result.insert(line[1..].to_string(), vec![]);
                continue;
            }
            if current_gene.is_none() {
                return Err(ParseError::new(format!(
                    "Expected #name line on line {} in file {}",
                    line_no + 1,
                    in_path
                ))
                .into());
            }
            let tokens: Vec<&str> = line.split(':').collect();
            let mut_type: MutationType = tokens[0].parse::<u8>()?.into();
            let probability: Float = tokens[1].parse()?;
            if let Some(gene) = &current_gene {
                result
                    .get_mut(gene)
                    .expect("init")
                    .push(MutationEvent::new(mut_type, probability));
            //result.entry(gene.clone()).or_insert_with(||Vec::new()).push(
            } else {
                panic!("gene must be Some because of previous check");
            }
        } else {
            Err(line.unwrap_err()).with_context(|| format!("IO error for file {}", in_path))?
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryInto;

    #[test]
    fn test_possible_mutations_io() {
        use mutexpect::MutationEvent;

        //external constructor
        fn mevent(mut_type: &str, probability: Float) -> MutationEvent {
            MutationEvent::new(mut_type.try_into().unwrap(), probability)
        }

        //this is not portable and will not work on Windows
        let path = "/tmp/unit_test.possible_mutations";
        let mut pm: PossibleMutations = HashMap::new();
        write_to_file(path, &pm).unwrap();
        let pm2 = read_from_file(path).unwrap();
        assert_eq!(pm, pm2);

        pm.insert("foo".to_string(), vec![]);
        write_to_file(path, &pm).unwrap();
        let pm2 = read_from_file(path).unwrap();
        assert_eq!(pm, pm2);

        let events = vec![
            mevent("synonymous", 0.1),
            mevent("missense", 0.2),
            mevent("nonsense", 0.3),
        ];
        pm.insert("bar".to_string(), events);

        write_to_file(path, &pm).unwrap();
        let pm2 = read_from_file(path).unwrap();
        assert_eq!(pm, pm2);

        pm.insert("baz".to_string(), vec![]);
        write_to_file(path, &pm).unwrap();
        let pm2 = read_from_file(path).unwrap();
        assert_eq!(pm, pm2);
    }
}
