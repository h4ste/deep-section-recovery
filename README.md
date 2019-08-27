# Deep Section Recovery Model (DSRM)
This repository contains the source code for the Deep Section Recovery Model (DSRM) described in [__Inferring Clinical Correlations from EEG Reports with Deep Neural Learning__](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5977577/) presented at the 2017 AMIA Annual Symposium.

![DSRM Thumbnail](https://github.com/h4ste/deep-section-recovery/raw/master/dsrm.png "Overview of the Deep Section Recovery Model (DSRM)")

## Overview
At a high level, the DSRM can be viewed as operating through two general steps:
1. word- and report- level features are automatically extracted from each EEG report to capture contextual, semantic, and
background knowledge; and
2. the most likely clinical correlation section is jointly (a) inferred and (b) expressed through automatically generated
natural language.

Illustrated above, these two steps correspond to the two major components of the DSRM:
- the Extractor which learns how to automatically extract (a) feature vectors representing contextual and background knowledge associated with each word in a given EEG report as well as (b) a feature vector encoding semantic, background, and domain knowledge about the entire report; and
- the Generator which learns how to use the feature vectors extracted by the Extractor to produce the most likely clinical
correlation section for the given report while also considering the semantics of the natural language it is generating.

## Usage

Coming soon!


