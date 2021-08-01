use anyhow::Result;
use clap::{App, Arg};

use pattern_partition_prediction::{PaPaPred, PaPaPredIndel};
use twobit::TwoBitFile;

mod compare;
mod counts;
mod enumerate;
mod error;
mod expect;
mod io;
mod observed;
mod sample;
mod transform;

const VERSION: &'static str = env!("CARGO_PKG_VERSION");

type Float = f32;
type MutationType = mutexpect::MutationType;

use crate::compare::compare_mutations;
use crate::enumerate::enumerate_possible_mutations;
use crate::error::MissingCommandLineArgumentError;
use crate::expect::expected_number_of_mutations;
use crate::observed::classify_mutations;
use crate::observed::read_mutations_from_file as read_observed_mutations_from_file;
use crate::sample::sample_mutations;

fn require_initialization<'a, T>(
    value: &'a Option<T>,
    cli_argument: &'static str,
) -> std::result::Result<&'a T, MissingCommandLineArgumentError> {
    match value {
        Some(v) => Ok(v),
        None => Err(MissingCommandLineArgumentError::new(cli_argument)),
    }
}

fn main() -> Result<()> {
    let app = App::new("genovo")
        .version(VERSION)
        .author("Jörn Bethune")
        .about("Determine genes enriched with de-novo mutations")
        .after_help("If no --action is given, all actions are executed.\n\
                     Possible actions are: transform, enumerate, expect, sample, classify, compare" )
        .arg(Arg::with_name("action")
             .long("action")
             .value_name("ACTION")
             .help("Only run a specific step in the pipeline")
             .takes_value(true))

        //raw data arguments
        .arg(Arg::with_name("gff3")
             .long("gff3")
             .value_name("FILE")
             .help("gff3 gene annotations file")
             .takes_value(true))
        .arg(Arg::with_name("genome")
             .long("genome")
             .value_name("FILE")
             .help("A 2bit reference genome sequence file")
             .takes_value(true))
        .arg(Arg::with_name("point-mutation-probabilities")
             .long("point-mutation-probabilities")
             .value_name("FILE")
             .help("A pattern partition prediction point mutation probability table")
             .takes_value(true))
        .arg(Arg::with_name("indel-mutation-probabilities")
             .long("indel-mutation-probabilities")
             .value_name("FILE")
             .help("A pattern partition prediction indel mutation probability table")
             .takes_value(true))
        .arg(Arg::with_name("observed-mutations")
             .long("observed-mutations")
             .value_name("FILE")
             .help("A vcf-like file containing observed point mutations")
             .takes_value(true))

        // input/output file arguments
        .arg(Arg::with_name("genomic-regions")
             .long("genomic-regions")
             .value_name("FILE")
             .help("Locations of exons, CDS and their phases for each gene")
             .takes_value(true))
        .arg(Arg::with_name("possible-mutations")
             .long("possible-mutations")
             .value_name("FILE")
             .help("A list of all possible point mutations for each gene")
             .takes_value(true))
        .arg(Arg::with_name("classified-mutations")
             .long("classified-mutations")
             .value_name("FILE")
             .help("Observed, classified point mutations")
             .takes_value(true))
        .arg(Arg::with_name("expected-mutations")
             .long("expected-mutations")
             .value_name("FILE")
             .help("Expected number of point mutations per gene")
             .takes_value(true))
        .arg(Arg::with_name("sampled-mutations")
             .long("sampled-mutations")
             .value_name("FILE")
             .help("Sampled number of point mutations per gene")
             .takes_value(true))
        .arg(Arg::with_name("significant-mutations")
             .long("significant-mutations")
             .value_name("FILE")
             .help("Statistical test results for every gene")
             .default_value("-")
             .takes_value(true))

        // non-file args
        .arg(Arg::with_name("id")
             .long("--id")
             .value_name("NAME")
             .help("Only process a gene/transcript with the given ID (useful for parallel processing)")
             .takes_value(true))
        .arg(Arg::with_name("scaling-factor")
             .long("--scaling-factor")
             .value_name("NAME")
             .help("Scaling factor for all mutation probabilities")
             .default_value("1.0")
             .takes_value(true))
        .arg(Arg::with_name("sum-up-observed-mutations-per-transcript")
             .long("--sum-up-observed-mutations-per-transcript")
             .help("Tally up the number of observed mutations instead of displaying them in detail (can be used in combination with --action classify and --classified-mutations <output-file>)"))
        .arg(Arg::with_name("include-intronic")
             .long("--include-intronic")
             .help("Also enumerate intronic variants. Default is to not enumerate intronic variants."))
        .arg(Arg::with_name("include-unknown")
             .long("--include-unknown")
             .help("Also enumerate variants with unknown type (primarily UTR). Default is to not enumerate variants with unknown type."))
        .arg(Arg::with_name("number-of-random-samples")
             .long("--number-of-random-samples")
             .value_name("NUMBER")
             .default_value("1000")
             .help("The number of random samples that should be generated")
             .takes_value(true))
        .arg(Arg::with_name("required-tag")
             .long("--required-tag")
             .value_name("TAG")
             .help("Exclude entrences from gff file that to no include the tag TAG")
             .takes_value(true)
             .multiple(true))
        .arg(Arg::with_name("filter-plof")
             .long("--filter-plof")
             .help("filter putatuve LoF variants. (50 bp rule)"))

    ;


    let matches = app.get_matches();
    let run_all = matches.value_of("action").is_none();
    let id = matches.value_of("id");
    let scaling_factor: f32 = matches
        .value_of("scaling-factor")
        .expect("default value")
        .parse()?;
    let include_intronic = matches.occurrences_of("include-intronic") > 0;
    let include_unknown = matches.occurrences_of("include-unknown") > 0;
    let filter_plof = matches.occurrences_of("filter_plof") > 0;
  
    let required_tags: Option<Vec<&str>> = {
       if let Some(tags) = matches.values_of("required-tag"){
           Some(tags.collect())
       } else {
           None
       }
    };
   
    //println!("{:#?}", required_tags);
    
    
    /*
     * Depending on what --action is given, each step may or may not produce results.
     * Therefore the variables are all Option's.
     */

    let ref_genome = {
        if let Some(ref_genome_file) = matches.value_of("genome") {
            Some(TwoBitFile::open(ref_genome_file, false)?)
        } else {
            None
        }
    };

    let papa = {
        if let Some(papa_file) = matches.value_of("point-mutation-probabilities") {
            Some(PaPaPred::new(papa_file, Some(5))?) // 5 to have at least 2 flanking bases around a point mutation to always have a full codon available for every coding site
        } else {
            None
        }
    };

    let papa_indel = {
        if let Some(papa_file) = matches.value_of("indel-mutation-probabilities") {
            let min_width = {
                if let Some(papa_point) = &papa {
                    Some(papa_point.kmer_size().saturating_sub(1))
                } else {
                    None
                }
            };
            Some(PaPaPredIndel::new(papa_file, min_width)?)
        } else {
            None
        }
    };

    let observed_mutations = {
        if let Some(observed_mutations_file) = matches.value_of("observed-mutations") {
            Some(read_observed_mutations_from_file(
                observed_mutations_file,
                -1,
            )?) //TODO expose adjustment parameter to CLI
        } else {
            None
        }
    };

    println!("transform");
    // action=transform
    let regions = {
        if let Some(gff3) = matches.value_of("gff3") {
            let regions = transform::transform_gff3_annotations(gff3, id, required_tags)?;
            if let Some(regions_file) = matches.value_of("genomic-regions") {
                transform::write_sequence_annotations_to_file(regions_file, &regions)?;
            }
            if !run_all {
                // we are done here
                return Ok(());
            }
            Some(regions)
        } else if let Some(regions_file) = matches.value_of("genomic-regions") {
            Some(mutexpect::read_sequence_annotations_from_file(
                regions_file,
                id,
            )?)
        } else if run_all || matches.value_of("action") == Some("transform") {
            {
                return Err(anyhow::anyhow!("Please provide the --gff3 parameter"));
            }
        } else {
            None
        }
    };

    println!("enumerate");
    //action=enumerate
    let possible_mutations = {
        if run_all || matches.value_of("action") == Some("enumerate") {
            let possible_mutations = enumerate_possible_mutations(
                require_initialization(&regions, "--genomic-regions")?,
                require_initialization(&ref_genome, "--genome")?,
                require_initialization(&papa, "--point-mutation-probabilities")?,
                &papa_indel,
                scaling_factor,
                true,
                id,
                include_intronic,
                include_unknown,
            )?;

            if let Some(possible_mutations_file) = matches.value_of("possible-mutations") {
                enumerate::write_to_file(possible_mutations_file, &possible_mutations)?
            }
            if !run_all {
                // we are done here
                return Ok(());
            }
            Some(possible_mutations)
        } else if let Some(possible_mutations_file) = matches.value_of("possible-mutations") {
            Some(enumerate::read_from_file(possible_mutations_file)?)
        } else {
            None
        }
    };

    println!("expect");
    //action=expect
    let expected_mutations = {
        if run_all || matches.value_of("action") == Some("expect") {
            let expected_mutations = expected_number_of_mutations(
                require_initialization(&possible_mutations, "--possible-mutations")?,
                id,
            )?;
            if let Some(expected_mutations_file) = matches.value_of("expected-mutations") {
                expect::write_to_file(expected_mutations_file, &expected_mutations)?;
            }
            if !run_all {
                // we are done here
                return Ok(());
            }
            Some(expected_mutations)
        } else if let Some(expected_mutations_file) = matches.value_of("expected-mutations") {
            Some(expect::read_from_file(expected_mutations_file)?)
        } else {
            None
        }
    };

    println!("sample");
    //action=sample
    let sampled_mutations = {
        if run_all || matches.value_of("action") == Some("sample") {
            let iterations: usize = matches
                .value_of("number-of-random-samples")
                .expect("clap default value")
                .parse()
                .unwrap(); //TODO proper error handling
            let sampled_mutations = sample_mutations(
                require_initialization(&possible_mutations, "--possible-mutations")?,
                iterations,
                id,
            )?;

            if let Some(sampled_mutations_file) = matches.value_of("sampled-mutations") {
                sample::write_to_file(sampled_mutations_file, &sampled_mutations)?;
            }
            if !run_all {
                // we are done here
                return Ok(());
            }
            Some(sampled_mutations)
        } else if let Some(sampled_mutations_file) = matches.value_of("sampled-mutations") {
            Some(sample::read_from_file(sampled_mutations_file)?)
        } else {
            None
        }
    };

    std::mem::drop(possible_mutations); // let's free up some memory

    println!("classify");
    //action=classify
    let classified_mutations = {
        if run_all || matches.value_of("action") == Some("classify") {
            let classified_mutations = classify_mutations(
                require_initialization(&observed_mutations, "--observed-mutations")?,
                require_initialization(&regions, "--genomic-regions")?,
                require_initialization(&ref_genome, "--genome")?,
                id,
            )?;

            if let Some(classified_mutations_file) = matches.value_of("classified-mutations") {
                if matches.occurrences_of("sum-up-observed-mutations-per-transcript") > 0 {
                    // sum up the number of mutations of each type for each transcript
                    observed::sum_up_and_write_to_file(
                        classified_mutations_file,
                        &classified_mutations,
                    )?;
                } else {
                    // Write the classification for each individual mutation on a separate line
                    // This is what you want if you want to run the full pipeline
                    observed::write_to_file(classified_mutations_file, &classified_mutations)?;
                }
            }
            if !run_all {
                // we are done here
                return Ok(());
            }
            Some(classified_mutations)
        } else if let Some(classified_mutations_file) = matches.value_of("classified-mutations") {
            Some(observed::read_from_file(classified_mutations_file)?)
        } else {
            None
        }
    };

    println!("compare");
    //action=compare
    let _significant_mutations = {
        if run_all || matches.value_of("action") == Some("compare") {
            let significant_mutations = compare_mutations(
                require_initialization(&classified_mutations, "--classified-mutations")?,
                require_initialization(&expected_mutations, "--expected-mutations")?,
                require_initialization(&sampled_mutations, "--sampled-mutations")?,
                id,
            )?;

            if let Some(significant_mutations_file) = matches.value_of("significant-mutations") {
                compare::write_to_file(significant_mutations_file, &significant_mutations)?;
            }
            if !run_all {
                // we are done here
                return Ok(());
            }
            Some(significant_mutations)
        } else {
            // last step in the pipeline. No work needs to be done
            None
        }
    };

    if !run_all {
        return Err(anyhow::anyhow!(
            "Invalid --action parameter: {}",
            matches.value_of("action").expect("clap")
        ));
    }
    Ok(())
}
