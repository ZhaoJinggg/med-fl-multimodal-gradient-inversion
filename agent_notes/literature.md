# Literature verification log (started 2026-11-22)

Running log; each question appended once verified. "not verified" = could not confirm from a primary source.

## Q4 (DAGER precision) -- local code, partial
- Local copy: third_party/dager, commit 5f8e306223ad35f3b8ce61b35f1ff60158207310 (2025-08-13), origin https://github.com/insait-institute/dager-gradient-inversion
- args_factory.py:31 `--precision` choices ['8bit','half','full','double'], default 'full'
- utils/models.py:31-32 `double` -> `torch_dtype = torch.float64` when loading the HF model
- scripts/fed_avg.sh:6 and scripts/fa.sh:6 (both `--algo fedavg`) pass `--precision double`
- Public repo HEAD (api.github.com/repos/insait-institute/dager-gradient-inversion/commits) = same commit 5f8e306 (6 commits total); raw master scripts/fed_avg.sh identical, ends `--precision double`. Commit 14a9056 "Fixed script" only touched gpt2-large.sh.
- No other script passes --precision -> all non-FedAvg experiments run default 'full' (float32).
- compute_grads_fed_avg (utils/models.py:123-146): client SGD simulated on the same self.model (float64 under double), update = -(w_after - w_before)/n_minib/lr/E. Also get_matrices_expansions adds B += 60 for fedavg (models.py:232-233).
- README: FedAvg command `./scripts/fed_avg.sh AVG_EPOCHS B_MINI LEARNING_RATE`, no mention of precision.
- Paper arXiv 2405.15586v2 (HTML): no "float32/float64/double precision/fp16/rounding". Only: Sec 5.1 "robust to numerical errors caused by floating-point arithmetic" (span check under "infinite precision" arithmetic, handled by thresholds); App. C "To manage numerical instabilities within the framework"; Sec 6.1 "16-bit quantization" (fine-tuning/LoRA experiments).
- Table 4 (GPT-2, RT, B=16; default E=10, lr=1e-4, Bmini=4): E=20 -> R1 96.0+-1.4; lr 5e-4 -> 84.2; App B.3 "key factor ... is observed to be the learning rate".
- VERDICT: paper's sentence "Its FedAvg experiment runs the client in double precision" is CONFIRMED (more precisely: the released FedAvg script loads the whole model, so both the simulated client training and the attack, in float64; the paper itself does not state this).

## Q1 (Yin et al. 2021, IIP) -- DONE
- Sources: arXiv 2104.07586v1 PDF (https://arxiv.org/pdf/2104.07586v1) and CVF open-access PDF (https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_See_Through_Gradients_Image_Batch_Recovery_via_GradInversion_CVPR_2021_paper.pdf); IIP text identical in both. Crossref: DOI 10.1109/CVPR46437.2021.01607, pp. 16332-16341 (IEEE Xplore pagination).
- Location: Sec. 4.3 "Effect of scaling up the batch size", paragraph "Image Identifiablity Precision (IIP)" [sic], and Fig. 8 caption.
- Definition: measures "how easy it is to identify a particular image, given only its reconstruction, among all its similar peers in the original dataset"; "we calculate the fraction of exact matches between an original image and the nearest neighbor to its reconstruction".
- Similarity/feature space (Fig. 8 caption): "Nearest neighbors measured in avgpool feature space cosine similarity" (network not named in the caption; the setting is ResNet-50 on ImageNet). Candidate set: "the original dataset"; Fig. 8 is on the ImageNet validation set; exact size of the search set not stated.
- Top-1 hit rate (binary per image), aggregated as a fraction; not rank-based. Each point averaged over 256 random samples (240 at batch size 48). Reported only as a curve vs batch size (Fig. 8), ~28% identified at batch size 48; "vulnerable population" (largest grad norm per class) nearly doubles IIP at B=48. No table reports IIP.
- Differences from A: (1) A is a normalized rank (1-(r-1)/(N-1)), graded, chance 0.5; IIP is top-1 exact-match fraction, chance ~1/N. (2) Similarity: SSIM in pixel space vs cosine similarity of avgpool (penultimate) CNN features. (3) Candidate set: A = 3,199 radiographs minus the other batch images, explicitly defined; IIP = "original dataset" (ImageNet val), not further specified. (4) The share with r_i=1 reported alongside A is IIP with SSIM/pixel space substituted for avgpool cosine.
- Paper sentence (main.tex l.490) essentially accurate but should name the feature space Yin et al. use (avgpool features, cosine) and say IIP is a fraction of exact top-1 matches. Proposed wording in final report.

## Q2 (Hatamizadeh et al., TMI 2023) -- DONE
- Source: published TMI version, local copy references/05_Hatamizadeh_TMI2023.pdf (IEEE Xplore, DOI 10.1109/TMI.2023.3239391, vol 42 no 7 pp 2044-2056). Public: https://doi.org/10.1109/TMI.2023.3239391 ; preprint arXiv 2202.06924.
- Metrics defined/reported:
  (a) SSIM between original and reconstruction (Sec. III-C, Fig. 6a/b/c, Fig. 7). Text: SSIM chosen as "more intuitive and interpretable" than RMSE/PSNR. PSNR/LPIPS not reported.
  (b) SSIM(original, prior) as reference/"lower bound": Sec. III-D: "The SSIM value between the prior ... and the original image is 0.37 (see Fig. 7)". Reconstruction with SSIM <= SSIM(original, prior) "could be considered 'safe'".
  (c) RDLV, Eq. (6): RDLV = [SSIM(T_i, I_i) - SSIM(T_i, P)] / SSIM(T_i, P), T_i training image, I_i reconstruction, P prior used in the attack. Sec. III-E, Figs. 8 (simple Gaussian DP, CXR and fundus) and 9 (DP-SGD). Computed on "reconstructions that achieve the highest SSIM with any of the original training images", bootstrap 95% CI.
  (d) IIP / k-IIP (identification measure): Sec. II-D def; Sec. III-G "Membership Inference Attack Data Leakage"; Fig. 11. Features: ImageNet-pretrained ResNet-18 (512-d, Sec. III-F), cosine similarity, KNN; IIP = m/N, m = number of unique exact matches. Modified for FL: pool = "entire training and validation images of all clients"; "A reconstruction is counted as an exact match if the closest match originates from the attacked client" (i.e. client-level hit, not image-level). Fig. 11 legend k = 1, 3, 5 (role of k not spelled out beyond KNN). Result: client 9 IIP 1.0 up to sigma0 = 20; other clients "significantly lower values of the IIP score (e.g., <0.1)". Also t-SNE of ResNet-18 features (Fig. 10).
- Prior image: mean image of ChestX-ray14 (Sec. III-A) for CXR (dataset A); mean ODIR-2019 image for fundus.
- "SSIM 0.64 (prior 0.37)": numbers are from Fig. 7 (image labels SSIM=0.64 for sigma=0, 0.26 for sigma=1e-2, prior 0.37), high-risk client 9 (one image, batch 1, one iteration), FL round 90. Pixel SSIM. ACCURATE but it is a single illustrated example. Fig. 6a/6b curves: BN loss + prior at bs=1 / 1 iteration ~0.7 / ~0.66 (read off plot), falling to ~0.45 / ~0.37.
- "no leakage beyond the prior for other clients": supported by Sec. III-E text ("clients with more than one training image do not leak data in this regime within the confidence interval as their reconstructions are less similar than the prior"), i.e. RDLV-based, in the DP sweep of Fig. 8. Nuance: Fig. 11 shows nonzero k-IIP (~0.1-0.25) for some other clients at sigma0=0.
- Q3(b)-relevant quotes: Fig. 6 caption "a SSIM value of ~0.5 still indicates no recognizable inversion result"; Sec. III-E absolute SSIM values across clients "incomparable" due to varying reference images; Limitations (Sec. IV): some reconstructions in positive-RDLV regime "have a low IIP score" yet "some qualities that cause a relatively high SSIM metric"; suggests deep-feature similarity instead.

## Q3 (precedent) -- DONE
(a) identification/retrieval metric:
 1. Hatamizadeh 2023 TMI (medical CXR, BN-statistics attack): IIP/k-IIP (Sec II-D, III-G, Fig 11), ResNet-18 features + cosine; client-level hit. Secondary to SSIM/RDLV, not primary.
 2. Yin 2021 CVPR (BN-statistics attack, ImageNet): IIP introduced (Sec 4.3, Fig 8); main comparison table (Table 4) uses FFT2D/PSNR/LPIPS, so IIP is not the primary measure there either.
 3. Fowl et al., Robbing the Fed, ICLR 2022 (arXiv 2110.13057v2, Sec 4): "measure PSNR scores as well as Image Identifiability Precision (IIP) scores"; "For IIP we search for nearest-neighbors in pixel space ... a more strict metric compared to IIP as used in Yin et al."; Fig 4 "Identification Success (via IIP score)". OUTSIDE requested scope: malicious-model threat model, ImageNet, not medical, not BN.
 4. Subbanna, Wilms, Tuladhar, Forkert, Sensors 21(11):3874, 2021, doi 10.3390/s21113874 (PMC8199987): MODEL inversion (latent-space access) on brain-MRI segmentation nets, not gradient inversion. Sec 4.2.1: reconstruction SSIM vs all 40 test subjects, rank of true original; U-Net highest in "30 of the 40 cases", mean rank 1.5; SSIM 0.73+-0.12 true vs 0.50+-0.23 others (Table 2). Closest precedent to a rank-by-SSIM identification score.
 Code precedent: Breaching (github.com/JonasGeiping/breaching, breaching/analysis/metrics.py) implements image_identifiability_precision "as described in Yin et al.": candidates = dataset images with the same label; scores pixel L2 / LPIPS / model features; top-1 argmin; returns fraction identified.
 Not found: any medical GI paper using identification as PRIMARY measure.
(b) SSIM vs prior/mean uninformative on CXR:
 - Hatamizadeh 2023: Fig 6 caption "a SSIM value of ~0.5 still indicates no recognizable inversion result"; Sec III-E absolute SSIM across clients "incomparable" because of reference images -> RDLV relative to prior; Sec IV limitations: reconstructions with positive RDLV had "a low IIP score" but "qualities that cause a relatively high SSIM metric"; suggests deep-feature metrics. Prior SSIM 0.37 (Fig 7).
 - Jiang 2025 MedIA (already cited): adopts RDLV, Eq 19, prior = "the mean of an image dataset"; RDLV<0 => risk "negligible". Does not argue SSIM is uninformative.
 - Not found: a paper that states explicitly that the mean CXR itself scores high SSIM because radiographs are mutually similar. Searched (web) without result -> "not verified".

## Q5 (CAR-MFL, Huang et al.) -- DONE
- Context: design_review_2026-11-21.md Sec 4 "Fusion mode": CAR-MFL = precedent for the architecture (two encoders, concatenation, one linear layer, FedAvg); Huang et al. 2020 = medical fusion taxonomy ("joint fusion").
- CAR-MFL: Poudel, Shrestha, Amgain, Shrestha (Yash Raj), Gyawali, Bhattarai. "CAR-MFL: Cross-Modal Augmentation by Retrieval for Multimodal Federated Learning with Missing Modalities". MICCAI 2024, LNCS 15010 (Part X; volume number from TUM library catalog record, Crossref gives Part X only), pp. 102-112, Springer, DOI 10.1007/978-3-031-72117-5_10; arXiv 2407.08648 ("Accepted at MICCAI 2024").
  Sec 2: shared architecture "modality-specific encoders", "a concatenation-based fusion module", "classifier head". Sec 3 Implementation: pretrained ResNet50 + BERT-base, outputs reduced to 256-d and L2-normalized before fusion, "simple concatenation approach ... followed by a linear layer"; Adam lr 1e-4, 3 local epochs/round, 30 rounds. Homogeneous setup 10 clients (MIMIC-CXR); heterogeneous: 8 CheXpert image-only + 2 Open-I multimodal clients. Baseline mFedAvg (FedAvg adapted to multimodal data). Multi-label classification, AUC.
  Caveat for design review: CAR-MFL is not plain FedAvg-trained concatenation model -- it's a missing-modality method built on FedAvg-style aggregation; also 256-d projection + L2 norm before concat.
- Huang, Pareek, Seyyedi, Banerjee, Lungren. "Fusion of medical imaging and electronic health records using deep learning: a systematic review and implementation guidelines". npj Digit. Med. 3, 136 (2020). DOI 10.1038/s41746-020-00341-z, PMC7567861.
  "Terminology and strategies in fusion", Fig 2: early fusion = "joining multiple input modalities into a single feature vector" (type II: extracted features, incl. learned representation, not updated); "Joint fusion (or intermediate fusion)" = "joining learned feature representations from intermediate layers of neural networks with features from other modalities", with "the loss is propagated back to the feature extracting neural networks during training"; late fusion = "leveraging predictions from multiple models to make a final decision". Note: domain is imaging + EHR, not image + report text.

## Q6 BibTeX -- DONE (see final report; keys yin2021gradinversion, fowl2022robbing, subbanna2021modelinversion, poudel2024carmfl, huang2020fusion)
- Fowl et al.: arXiv 2110.13057v2 header "Published as a conference paper at ICLR 2022"; authors as printed: Liam Fowl, Jonas Geiping, Wojtek Czaja, Micah Goldblum, Tom Goldstein. OpenReview page blocked (bot check) -> forum id not verified.
- DAGER table rows checked: BERT RT B=8 R1 37.1 OK; GPT-2 RT B=16 100.0 OK; 4-27 words OK; ECHR 512 = truncation (dataset has >1000-word docs).

## Status: ALL QUESTIONS DONE 2026-11-22
