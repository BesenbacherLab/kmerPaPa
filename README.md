# kmerPaPa
Tool to calculate a "k-mer pattern partition" from position specific k-mer counts. This can for instance be used to train a mutation rate model.

## Requirements
kmerPaPa requires Python 3.8 or above.

## Installation
kmerPaPa can be installed using pip:
```
pip install kmerpapa
```
or using [pipx](https://pypa.github.io/pipx/):
```
pipx install kmerpapa
```

## Test data
The test data files used in the usage examples below can be downloaded from the test_data directory in the project's github repository:
```
wget https://github.com/BesenbacherLab/kmerPaPa/raw/main/test_data/mutated_5mers.txt
wget https://github.com/BesenbacherLab/kmerPaPa/raw/main/test_data/background_5mers.txt
```

## Usage
If we want to train a mutation rate model then the input data should specifiy the number of times each k-mer is observed mutated and unmutated. One option is to have one file with the mutated k-mer counts (positive) and one file with the count of k-mers in the whole genome (background).  We can then run kmerpapa like this:
```
kmerpapa --positive mutated_5mers.txt \
         --background background_5mers.txt \
         --penalty_values 3 5 7
```
The above command will first use cross validation to find the best penalty value between the values 3,5 and 7. Then it will find the optimal k-mer patter partiton using that penalty value.
If both a list of penalty values and a list of pseudo-counts are specified then all combinations of values will be tested during cross validation:
```
kmerpapa --positive mutated_5mers.txt \
         --background background_5mers.txt \
         --penalty_values 3 5 6 \
         --pseudo_counts 0.5 1 10
```
If only a single combination of penalty_value and pseudo_count is provided then the default is not to run cross validation unless "--n_folds" option or the "CV_only" is used. The "CV_only" option can be used together with "--CVfile" option to parallelize grid search.
Fx. using bash:
```
for c in 3 5 6; do
    for a in 0.5 1 10; do
        kmerpapa --positive mutated_5mers.txt \
         --background background_5mers.txt \
         --penalty_values $c \
         --pseudo_counts $a \
         --CV_only --CVfile CV_results_c${c}_a${a}.txt &
    done
done
```

## Creating input data
Input files with k-mer counts can be created using [kmer_counter](https://github.com/BesenbacherLab/kmer_counter).
Given a file of point mutations in a file that contain the CHROM, POS, REF and ALT columns from a vcf file:
```
chr1 1000000 G A
chr1 1000100 G A
chr1 1000200 C T
chr1 1000300 C T
chr1 1000400 C T
```
We can count the 5-mers around each mutation using this command:
```
kmer_counter snv --radius 2 {genome}.2bit {point_mutations_file} > mutated_5mers.txt
```
Given a bed file with regions that are sufficiently covered by sequencing we can count the background 5-mers using this command:
```
kmer_counter background --bed {regions}.bed -radius 2 {genome}.2bit > background_5mers.txt
```

The file `{genome}.2bit` should be a 2bit file of the same reference genome that were used for calling the mutations. 2bit files can be downloaded from: `https://hgdownload.cse.ucsc.edu/goldenpath/{genome}/bigZips/{genome}.2bit` where `{genome}` is a valid UCSC genome assembly name (fx. "hg38").



