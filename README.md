# Controlled Transformation of Text-Attributed Graphs
Graph generation is the process of generating novel graphs with similar attributes to real world graphs. The explicit and precise control of granular structural attributes, such as node centrality and graph density, is crucial for effective graph generation. This paper introduces a controllable multi-objective translation model for text-attributed graphs,titled Controlled Graph Translator (CGT). It is designed to effectively and efficiently translate a given source graph to a target graph, while satisfying multiple desired graph attributes at granular level. Designed with an encoderdecoder architecture, CGT develops fusion and graph attribute predictor neural networks for controlled graph translation. We validate the effectiveness of CGT through extensive experiments on different genres of datasets. In addition, we illustrate the application of CGT in data augmentation and taxonomy creation, particularly in low resource settings.

<p align="center">
<img src="https://github.com/nidhivakil/CGT/blob/main/image/controlled_graph_transformer_CGT.drawio.png" width="900" height="450">
</p>

# Data 

There are five datasets: 
* Arxiv
* Wordnet
* Mutag
* Molbace
* Citeseer
  
To download datasets and Train/Test/Val splits, go to data directory and run download.sh as follows

```
sh data/download.sh
```

# To run the code 
Use the following command with appropriate arguments:
```
python cgg_cnn_v3_train_model.py
```

# Citation

```
@inproceedings{vakil-amiri-2024-controlled,
    title = "Controlled Transformation of Text-Attributed Graphs",
    author = "Vakil, Nidhi  and
      Amiri, Hadi",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.923/",
    doi = "10.18653/v1/2024.findings-emnlp.923",
    pages = "15735--15748"
}
```
