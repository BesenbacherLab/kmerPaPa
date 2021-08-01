use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize, Serializer, Deserializer};
use serde::de::{Visitor, Unexpected};

use mutexpect::interval::Interval;
use mutexpect::{MutationType, PointMutationClassifier};
use tabfile::Tabfile;
use twobit::TwoBitFile;

use crate::compare::tally_up_observed_mutations;
use crate::counts::ObservedMutationCounts;
use crate::error::ParseError;
use crate::io::{get_reader, get_writer};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Mutation {
    pub region: Option<String>,
    pub chromosome: String,
    pub position: usize,
    pub mutation_type: MutationType,
    pub change: Change,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Change {
    PointMutation(char, char),
    Indel(String, String)
}

impl Mutation {
    pub fn new(region: Option<String>, chromosome: String, position: usize, from: String, to: String) -> Self {
        let change = if from.len() == 1 && to.len() == 1 { // point mutation
            Change::PointMutation(
                from.chars().next().expect("length"),
                to.chars().next().expect("length")
            )
        } else {
            if from.len() == 0 || to.len() == 0 {
                panic!("Encountered indel without anchor base: {} -> {}", from, to);
            }
            Change::Indel( from, to )
        };
        Self {region, chromosome, position, change, mutation_type: MutationType::Unknown, }
    }

    pub fn ref_base(&self) -> char {
        match self.change {
            Change::PointMutation(n, _) => n,
            Change::Indel(ref from, _ ) => from.chars().next().expect("already sanitized"),
        }
    }

    pub fn alt_base(&self) -> Option<char> {
        if let Change::PointMutation(_, to) = self.change {
            Some(to)
        } else {
            None
        }
    }
}

impl Change {
    pub fn is_frameshift(&self) -> bool {
        match self {
            Change::PointMutation(_, _) => false,
            Change::Indel(inserted, deleted) => {
                let net_length: isize = inserted.len() as isize - deleted.len() as isize;
                net_length.rem_euclid(3) != 0
            }
        }
    }
}

// We need to flatten our tuple variants, so that the CSV crate can work with them
impl Serialize for Change {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut change = String::with_capacity(32);
        match *self {
            Change::PointMutation(ref from, ref to) => {
                change.push(*from);
                change.push_str("->");
                change.push(*to);
            }
            Change::Indel(ref from, ref to) => {
                change.push_str(from);
                change.push_str("->");
                change.push_str(to);
            }
        }
        serializer.serialize_str(&change)
    }
}

impl<'de> Deserialize<'de> for Change {
    fn deserialize<D>(deserializer: D) -> Result<Change, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(deserializer.deserialize_str(ChangeVisitor)?)
    }
}

struct ChangeVisitor;

impl<'de> Visitor<'de> for ChangeVisitor {
    type Value = Change;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a string of the form ref->alt")
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where E: serde::de::Error,
    {
        let parts: Vec<&str> = v.split("->").collect();
        if parts.len() != 2 {
            return Err(serde::de::Error::invalid_value(Unexpected::Str(v), &self))
        }

        if parts[0].len() > 1 || parts[1].len() > 1 { // indel
            Ok(Change::Indel(parts[0].to_string(), parts[1].to_string()))
        } else {
            let from = parts[0].chars().next().expect("length");
            let to = parts[1].chars().next().expect("length");
            Ok(Change::PointMutation(from, to))
        }
    }
}


pub fn classify_mutations(
    observed_mutations: &[Mutation],
    annotations: &[mutexpect::SeqAnnotation],
    genome: &TwoBitFile,
    filter_for_id: Option<&str>,
    filter_plof : bool,
) -> Result<Vec<Mutation>> {
    let mut result = Vec::new();

    let flank = 2; // number of flanking bases left and right needed to classify all coding point mutations

    for annotation in annotations {
        if let Some(id) = filter_for_id {
            if id != annotation.name {
                continue;
            }
        }

        let seq_of_region: Vec<char> = genome
            .sequence(
                &annotation.chr,
                annotation.range.start - flank,
                annotation.range.stop + flank,
            )?
            .chars()
            .collect();
        assert_eq!(seq_of_region.len(), 2 * flank + annotation.range.len());

        let classifier = PointMutationClassifier::new(&annotation, 2);
        let mut relevant_mutations =
            filter_observed_mutations(&observed_mutations, &annotation.chr, annotation.range);
        for mutation in &mut relevant_mutations {
            let sequence_context: Vec<char> = {
                assert!(annotation.range.start <= mutation.position);
                let middle = mutation.position - annotation.range.start + flank;
                seq_of_region[middle - flank..middle + flank + 1].into()
            };
            assert_eq!(sequence_context[2], mutation.ref_base(), 
                        "Reference base in mutation file ({}) does not match base in 2bit file ({}). Are you using the right reference genome?",
                        mutation.ref_base(), sequence_context[2]); // sanity-check right reference genome

            let overlapping_intron = annotation.find_intron(mutation.position);

            let mut classified_mutation = mutation.clone();
            classified_mutation.region = Some(annotation.name.clone());
            match mutation.change {
                Change::PointMutation(_, _) => {
                    let mut mutation_type = classifier.classify_by_position(
                        mutation.position,
                        &sequence_context,
                        &overlapping_intron, // may be none
                    );

                    if mutation_type == MutationType::Unknown {
                        if let Some(overlapping_cds) = annotation.find_cds(mutation.position) {
                            mutation_type = classifier.classify_coding_mutation(
                                mutation.position,
                                &sequence_context,
                                mutation.alt_base().expect("point mutation"),
                                &overlapping_cds,
                                filter_plof,
                            );
                        }
                    }
                    classified_mutation.mutation_type = mutation_type;
                },
                Change::Indel(_, _) => {
                    let mutation_type = if let Some(_overlapping_cds) = annotation.find_cds(mutation.position + 1 ) { // +1 to ignore anchor base
                        if classified_mutation.change.is_frameshift() {
                            MutationType::FrameshiftIndel
                        } else {
                            MutationType::InFrameIndel
                        }
                    } else {
                        MutationType::Intronic
                    };
                    classified_mutation.mutation_type = mutation_type;
                },
            }
            result.push(classified_mutation);
        }
    }
    Ok(result)
}

pub fn read_mutations_from_file<P: AsRef<Path>>(
    file: P,
    adjust: i64,
) -> Result<Vec<Mutation>> {
    let mut result = Vec::new();

    let tsv_reader = Tabfile::open(&file)
        .with_context(|| format!("failed to open file {}", &file.as_ref().display()))?
        .comment_character('#')
        .separator(' ');

    for record_result in tsv_reader {
        let record = record_result?;
        let fields = record.fields();
        if fields.len() < 4 {
            return Err( ParseError::new(format!("Bad format in line {}. Expecting at least 4 instead of {} tab-delimited fields: chr, pos, ref, alt", record.line_number(), fields.len()) ).into());
        }
        let chromosome = fields[0].to_string();
        let position = {
            let value = fields[1].parse::<i64>()?;
            (value + adjust) as usize
        };

        let from;
        let to;
        from = fields[2];
        to = fields[3];
        result.push(Mutation::new(
            None,
            chromosome,
            position,
            from.to_string(),
            to.to_string(),
        ));
    }

    Ok(result)
}

fn filter_observed_mutations<'a>(
    mutations: &'a [Mutation],
    chr: &str,
    genomic_region: Interval,
) -> Vec<&'a Mutation> {
    let mut result = Vec::new();
    for mutation in mutations {
        // I assume no particular ordering. Otherwise a binary search might be faster
        if mutation.chromosome == chr && genomic_region.contains(mutation.position) {
            result.push(mutation)
        }
    }
    result
}

// serialization stuff //

pub fn write_to_file(out_path: &str, annotated_mutations: &[Mutation]) -> Result<()> {
    let writer = get_writer(out_path)
        .with_context(|| format!("failed to open file {} for writing", out_path))?;
    let mut csv_writer = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_writer(writer);
    for mutation in annotated_mutations {
        csv_writer.serialize(mutation)?;
    }
    Ok(())
}

/// Write a file with the following format:
/// ```
/// transcript_id<tab>synonymous<tab>missense<tab>...
/// ENSG1234<tab>synonymous_mutation_count<tab>missense_mutation_count<tab>...
/// ```
///
pub fn sum_up_and_write_to_file(
    out_path: &str,
    annotated_mutations: &[Mutation],
) -> Result<()> {
    let transcript_mutation_counts = tally_up_observed_mutations(annotated_mutations, None);

    let writer = get_writer(out_path)
        .with_context(|| format!("failed to open file {} for writing", out_path))?;
    let mut buf_writer = BufWriter::new(writer);
    buf_writer.write_all(b"name")?;
    for mut_type in ObservedMutationCounts::mutation_types() {
        buf_writer.write_fmt(format_args!("\t{}", mut_type))?;
    }
    buf_writer.write_all(b"\n")?; // end of header line

    // for every transcript
    for (name, counts) in transcript_mutation_counts {
        buf_writer.write(name.as_bytes())?;
        for count in counts {
            buf_writer.write_fmt(format_args!("\t{}", count))?;
        }
        buf_writer.write_all(b"\n")?;
    }
    Ok(())
}

pub fn read_from_file(in_path: &str) -> Result<Vec<Mutation>> {
    let mut result = Vec::new();
    let reader = get_reader(in_path)
        .with_context(|| format!("failed to open file {} for reading", in_path))?;
    let mut csv_reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_reader(reader);
    for row_result in csv_reader.deserialize() {
        let mutation: Mutation = row_result?;
        result.push(mutation);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observed_mutations_io() {
        let path = "/tmp/unit_test.observed_mutations";
        let mut om = Vec::new();
        write_to_file(path, &om).unwrap();
        let om2 = read_from_file(path).unwrap();
        assert_eq!(om2.len(), 0);

        om.push(Mutation {
            region: Some("my_gene".to_string()),
            mutation_type: MutationType::Synonymous,
            chromosome: "chr42".to_string(),
            position: 42,
            change: Change::PointMutation('A', 'T'),
        });
        write_to_file(path, &om).unwrap();
        let om2 = read_from_file(path).unwrap();
        assert_eq!(om, om2);

        om.push(Mutation {
            region: Some("my_other_gene".to_string()),
            mutation_type: MutationType::Nonsense,
            chromosome: "chrM".to_string(),
            position: 4,
            change: Change::PointMutation('C', 'G'),
        });
        write_to_file(path, &om).unwrap();
        let om2 = read_from_file(path).unwrap();
        assert_eq!(om, om2);

        om.push(Mutation::new(None, "chrM".to_string(), 4, "ACGT".to_string(), "A".to_string()));
        write_to_file(path, &om).unwrap();
        let om2 = read_from_file(path).unwrap();
        assert_eq!(om, om2);
    }
}
