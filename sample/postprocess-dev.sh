#/bin/sh

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=../../mosesdecoder

# suffix of target language files
lng=de

sed 's/\@\@ //g' | \
$mosesdecoder/scripts/recaser/detruecase.perl
