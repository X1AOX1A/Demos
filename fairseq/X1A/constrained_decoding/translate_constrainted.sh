# cd /path/to/download
# wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz
# tar -xzvf wmt19.de-en.joined-dict.ensemble.tar.gz

echo -e "Die maschinelle Übersetzung ist schwer zu kontrollieren.\thard\ttoinfluence" \
| /root/Documents/DEMOS/fairseq/examples/constrained_decoding/normalize.py \
| /root/Documents/DEMOS/fairseq/examples/constrained_decoding/tok.py \
| fairseq-interactive /root/Documents/MODELS/WMT_19/de-en/wmt19.de-en.joined-dict.ensemble \
  --path /root/Documents/MODELS/WMT_19/de-en/wmt19.de-en.joined-dict.ensemble/model1.pt \
  --bpe fastbpe \
  --bpe-codes /root/Documents/MODELS/WMT_19/de-en/wmt19.de-en.joined-dict.ensemble/bpecodes \
  --constraints \
  -s de -t en \
  --beam 10

# Output:
# S-0     Die masch@@ in@@ elle Über@@ setzung ist schwer zu kontrollieren .
# W-0     4.163   seconds
# C-0     hard
# C-0     to@@ influence
# H-0     -2.8746891021728516     Mach@@ ine trans@@ lation is hard to to@@ influence .
# D-0     -2.8746891021728516     Machine translation is hard to toinfluence .
# P-0     -0.5434 -0.1422 -0.1929 -0.1415 -0.2346 -1.8031 -0.1701 -17.1864 -10.8771 -0.1797 -0.1506
# 2023-05-07 14:36:12 | INFO | fairseq_cli.interactive | Total time: 15.882 seconds; translation time: 4.163