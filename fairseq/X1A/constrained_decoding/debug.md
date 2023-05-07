# Debug with fairseq-interactive

```bash
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
```

等价于先对输入进行 normalize 和 tokenize，再通过冒冷汗输入到 `fairseq-interactive` 中

```bash
python /root/Documents/DEMOS/fairseq/fairseq_cli/interactive.py /root/Documents/MODELS/WMT_19/de-en/wmt19.de-en.joined-dict.ensemble \
  --path /root/Documents/MODELS/WMT_19/de-en/wmt19.de-en.joined-dict.ensemble/model1.pt \
  --bpe fastbpe \
  --bpe-codes /root/Documents/MODELS/WMT_19/de-en/wmt19.de-en.joined-dict.ensemble/bpecodes \
  --constraints \
  -s de -t en \
  --beam 10

# Die maschinelle Übersetzung ist schwer zu kontrollieren .	hard 	toinfluence
```

debug: `interactive.py` with `launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug interactive.py",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "args": [
                "/root/Documents/MODELS/WMT_19/de-en/wmt19.de-en.joined-dict.ensemble",
                "--path",
                "/root/Documents/MODELS/WMT_19/de-en/wmt19.de-en.joined-dict.ensemble/model1.pt",
                "--bpe",
                "fastbpe",
                "--bpe-codes",
                "/root/Documents/MODELS/WMT_19/de-en/wmt19.de-en.joined-dict.ensemble/bpecodes",
                "--constraints",
                "-s",
                "de",
                "-t",
                "en",
                "--beam",
                "10",
            ],
            "console": "integratedTerminal"
        }
    ]
}

```

