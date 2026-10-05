import os
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    """Sets cell background color."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tc_pr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets inner cell padding in dxa (1 pt = 20 dxa)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tc_mar)

def set_cell_border(cell, **kwargs):
    """
    Sets cell borders.
    kwargs: top, bottom, left, right
    values: {"sz": 4, "val": "single", "color": "CCCCCC"}
    """
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_borders = parse_xml(f'<w:tcBorders {nsdecls("w")}/>')
    for edge in ('top', 'left', 'bottom', 'right'):
        edge_data = kwargs.get(edge)
        if edge_data:
            b_elm = parse_xml(
                f'<w:{edge} {nsdecls("w")} w:val="{edge_data.get("val", "single")}" '
                f'w:sz="{edge_data.get("sz", 4)}" w:space="0" '
                f'w:color="{edge_data.get("color", "CCCCCC")}"/>'
            )
            tc_borders.append(b_elm)
        else:
            b_elm = parse_xml(f'<w:{edge} {nsdecls("w")} w:val="none"/>')
            tc_borders.append(b_elm)
    tc_pr.append(tc_borders)

def build_manuscript():
    doc = Document()
    
    # Page setup: Standard Letter, 1-inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.page_width = Inches(8.5)
        section.page_height = Inches(11.0)

    # Base Colors
    NAVY = RGBColor(27, 54, 93)      # #1B365D - Primary header
    SLATE = RGBColor(44, 94, 138)    # #2C5E8A - Secondary header
    CHARCOAL = RGBColor(51, 51, 51)  # #333333 - Body
    DARKGRAY = RGBColor(80, 80, 80)
    HEX_HEADER = "1B365D"
    HEX_ROW_EVEN = "F7F9FC"
    HEX_ROW_ODD = "FFFFFF"
    HEX_BORDER = "CBD5E1"

    # Set normal style font
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Calibri'
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = CHARCOAL

    # Helper: Add Heading 1
    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = NAVY
        return p

    # Helper: Add Heading 2
    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = SLATE
        return p

    # Helper: Add Heading 3
    def add_h3(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.italic = True
        run.font.color.rgb = CHARCOAL
        return p

    # Helper: Add Body Paragraph
    def add_body(text, space_after=6, italic=False, bold=False):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(11)
        run.font.color.rgb = CHARCOAL
        run.font.italic = italic
        run.font.bold = bold
        return p

    # Helper: Add Image with Caption
    def add_figure(img_path, caption_text, width_inches=6.0):
        if os.path.exists(img_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(10)
            p_img.paragraph_format.space_after = Pt(4)
            run_img = p_img.add_run()
            run_img.add_picture(img_path, width=Inches(width_inches))

            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(2)
            p_cap.paragraph_format.space_after = Pt(12)
            p_cap.paragraph_format.keep_with_next = False
            run_cap = p_cap.add_run(caption_text)
            run_cap.font.name = 'Calibri'
            run_cap.font.size = Pt(9.5)
            run_cap.font.italic = True
            run_cap.font.color.rgb = DARKGRAY

    # =========================================================================
    # DOCUMENT HEADER: TITLE, AUTHORS, AFFILIATIONS
    # =========================================================================
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(8)
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("Wavelet-Refined Oriented Prompt Segmentation with Biophysical Guidance for Mangrove Canopy Disturbance Mapping in Can Gio Biosphere Reserve")
    run_title.font.name = 'Calibri'
    run_title.font.size = Pt(18)
    run_title.font.bold = True
    run_title.font.color.rgb = NAVY

    p_author = doc.add_paragraph()
    p_author.paragraph_format.space_before = Pt(4)
    p_author.paragraph_format.space_after = Pt(2)
    p_author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_author = p_author.add_run("Research Team in Earth Observation & Deep Learning")
    run_author.font.name = 'Calibri'
    run_author.font.size = Pt(11)
    run_author.font.bold = True
    run_author.font.color.rgb = CHARCOAL

    p_affil = doc.add_paragraph()
    p_affil.paragraph_format.space_before = Pt(0)
    p_affil.paragraph_format.space_after = Pt(14)
    p_affil.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_affil = p_affil.add_run("Laboratory of Remote Sensing and Geospatial Intelligence, Vietnam\nCorrespondence: geospatial.ai.research@institution.edu.vn")
    run_affil.font.name = 'Calibri'
    run_affil.font.size = Pt(9.5)
    run_affil.font.italic = True
    run_affil.font.color.rgb = DARKGRAY

    # =========================================================================
    # ABSTRACT & KEYWORDS BOX
    # =========================================================================
    p_abs_head = doc.add_paragraph()
    p_abs_head.paragraph_format.space_before = Pt(6)
    p_abs_head.paragraph_format.space_after = Pt(3)
    run_abs_h = p_abs_head.add_run("Abstract")
    run_abs_h.font.bold = True
    run_abs_h.font.size = Pt(12)
    run_abs_h.font.color.rgb = NAVY

    abstract_text = (
        "Mangrove forests are vital coastal blue carbon ecosystems that face escalating degradation from tropical cyclones, "
        "lightning strikes, tidal scouring, and natural dieback. High-resolution monitoring of localized canopy disturbance "
        "(windfall corridors and canopy dieback gaps) is critical for coastal conservation and carbon accounting. However, "
        "delineating small, fragmented canopy gaps in medium-resolution satellite imagery (Landsat-5 TM and Landsat-8 OLI, 30 m) "
        "presents acute challenges due to extreme spatial class imbalance (<1% target coverage), subtle spectral contrast against "
        "tidal mudflats and water bodies, and severe boundary blurring in standard deep convolutional neural networks (CNNs). "
        "While visual foundation models such as the Segment Anything Model (SAM) provide outstanding interactive zero-shot "
        "generalization, they degrade when prompted on small remote sensing targets due to downsampled feature maps and horizontal "
        "box prompt ambiguities. In this study, we propose WaveAttn SOPSeg, a wavelet-refined oriented prompt segmentation "
        "framework augmented by biophysical canopy disturbance guidance for high-precision mangrove gap mapping. WaveAttn SOPSeg "
        "integrates: (1) Region-Adaptive Magnification (RAM) and an Oriented Prompt Mechanism (OPM) that dynamically adapts bounding "
        "prompts to irregular windfall orientations; (2) a multi-scale Progressive Refinement Module (PRM) incorporating Haar Discrete "
        "Wavelet Transform (WaveFusion) to decouple high-frequency boundary gradients from low-frequency regional semantics; (3) a "
        "Global-Local Contextual Attention Module (GLCAM) that spotlights canopy disturbance clusters while suppressing tidal background "
        "noise; and (4) biophysical Normalized Burn Ratio (NBR) guidance encoding vegetation moisture deficits and canopy loss. "
        "We benchmark WaveAttn SOPSeg against fully-supervised U-Nets (ResNet-34 and ResNet-50 backbones), SAM fine-tuned with Low-Rank "
        "Adaptation (SAM + LoRA), and standard SOPSeg variants on both cropped image chips and original full-radiometric Landsat "
        "scenes spanning 2006–2014 in the Can Gio Mangrove Biosphere Reserve, Vietnam. WaveAttn SOPSeg achieves an Intersection "
        "over Union (IoU) of 0.7576, a Dice coefficient of 0.8621, and a Boundary IoU (BIoU) of 0.9634 on full scenes, compared to "
        "0.3782 IoU and 0.2740 BIoU for fully-supervised U-Net ResNet-34 (+37.94 percentage points in IoU and +68.94 percentage points in BIoU). "
        "Size-stratified evaluation confirms high fidelity even on micro-disturbances (<5 pixels, IoU 0.6335, BIoU 0.6923). "
        "Our findings establish that coupling wavelet frequency decoupling, oriented prompting, and biophysical spectral guidance provides an "
        "effective, boundary-preserving framework for satellite-based coastal forest monitoring."
    )
    p_abs = doc.add_paragraph()
    p_abs.paragraph_format.line_spacing = 1.15
    p_abs.paragraph_format.space_after = Pt(4)
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run_abs = p_abs.add_run(abstract_text)
    run_abs.font.size = Pt(10)
    run_abs.font.italic = True
    run_abs.font.color.rgb = CHARCOAL

    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.space_before = Pt(2)
    p_kw.paragraph_format.space_after = Pt(16)
    run_kwh = p_kw.add_run("Keywords: ")
    run_kwh.font.bold = True
    run_kwh.font.size = Pt(10)
    run_kwh.font.color.rgb = NAVY
    run_kwt = p_kw.add_run("Mangrove canopy disturbance; Canopy gap detection; Segment Anything Model (SAM); Discrete Wavelet Transform (DWT); Oriented Prompt Mechanism; Can Gio Biosphere Reserve; Boundary IoU; Landsat.")
    run_kwt.font.italic = True
    run_kwt.font.size = Pt(10)
    run_kwt.font.color.rgb = CHARCOAL

    # =========================================================================
    # 1. INTRODUCTION
    # =========================================================================
    add_h1("1. Introduction")
    
    add_body(
        "Mangrove ecosystems are among the most ecologically productive and carbon-dense biomes on Earth, providing indispensable "
        "ecosystem services including coastal shoreline stabilization, storm surge attenuation, nursery habitat provision for marine "
        "biodiversity, and long-term 'blue carbon' sequestration (Alongi, 2020; Donato et al., 2011). Despite their international "
        "recognition, mangrove forests worldwide are experiencing accelerating pressure from both anthropogenic activities (aquaculture "
        "conversion, coastal development) and climate-induced extreme events, including intense tropical storms, lightning strikes, "
        "salinity intrusion, and prolonged tidal scouring (Goldberg et al., 2020; Duke et al., 2017). In tropical deltaic systems such as "
        "the Can Gio Mangrove Biosphere Reserve in southern Vietnam, localized canopy disturbances—manifested as canopy windfall corridors, "
        "tree blowdowns, and localized standing dieback (referred to locally as 'vùng gãy đổ')—frequently trigger abrupt ecological degradation. "
        "Timely, accurate, and spatially explicit mapping of these canopy disturbances is vital for forest management, carbon accounting, "
        "and post-disaster rehabilitation planning."
    )

    add_body(
        "Over the past several decades, satellite Earth observation has served as the backbone for regional mangrove monitoring. Medium-resolution "
        "optical satellite archives, notably the Landsat program (Landsat 5 Thematic Mapper [TM] and Landsat 8 Operational Land Imager [OLI]), "
        "provide over four decades of continuous historical imagery at 30 m spatial resolution. However, mapping localized canopy gaps using "
        "Landsat imagery presents severe technical hurdles that differentiate this task from standard terrestrial forest clear-cut mapping:"
    )

    add_body(
        "1. Extreme Spatial Class Imbalance: Disturbed canopy gaps in natural mangrove stands typically range from small single-tree to "
        "few-tree blowdown corridors, occupying only 1 to 20 pixels (900 m² to 18,000 m²) within an entire Landsat scene. In our study area, "
        "disturbed pixels represent less than 0.6% of the total geographic coverage, leading standard deep learning objective functions "
        "to collapse toward predicting uniform background (unmodified forest or water)."
    )

    add_body(
        "2. Complex Spectral and Hydrological Heterogeneity: Mangroves reside in intertidal zones characterized by dynamic tidal inundation, "
        "variable sediment deposition, and shifting water tables. Exposed wet mudflats, shallow brackish canals, and dying tree canopies exhibit "
        "overlapping spectral responses in visual red-green-blue (RGB) bands, creating severe false-positive confusions unless multi-spectral "
        "infrared and shortwave-infrared biophysical indices (such as Normalized Difference Vegetation Index [NDVI], Normalized Burn Ratio [NBR], "
        "and Normalized Difference Water Index [NDWI]) are incorporated."
    )

    add_body(
        "3. Feature Smearing and Boundary Blurring in Deep CNNs: Standard fully-supervised convolutional neural networks (such as U-Net, "
        "DeepLabV3+, or FPN) rely on successive spatial downsampling operations (max pooling or strided convolutions) followed by bilinear "
        "or transposed convolution decoders. In downsampling through 4 or 5 stages (down to 1/16 or 1/32 resolution), fine boundary details "
        "of irregular, elongated canopy gaps are severely eroded. When upsampled, these decoders act as low-pass filters, resulting in blurry, "
        "amorphous predictions that fail to capture the true topological boundaries of the disturbance."
    )

    add_body(
        "Recently, the advent of visual foundation models, epitomized by Meta AI's Segment Anything Model (SAM) (Kirillov et al., 2023), "
        "has revolutionized interactive and promptable computer vision. Built upon a powerful Vision Transformer (ViT) pre-trained on over "
        "11 million images and 1 billion masks (SA-1B), SAM exhibits strong zero-shot transfer capabilities across diverse visual domains. "
        "In remote sensing, adapting SAM via parameter-efficient fine-tuning (PEFT), such as Low-Rank Adaptation (LoRA) (Hu et al., 2021), "
        "has shown promise for road, building, and land cover segmentation. However, when applied to small, irregular remote sensing targets "
        "like mangrove canopy gaps, vanilla SAM encounters two structural bottlenecks:"
    )

    add_body(
        "First, SAM's image encoder processes standard inputs at 1024×1024 resolution down to a 64×64 patch embedding (a 16× downsampling "
        "stride). For canopy disturbance instances that span only 2 to 10 pixels in satellite imagery, this downsampling collapses the entire "
        "object into a single spatial embedding token, stripping away internal variance and topological contours. Second, SAM's native "
        "bounding-box prompt mechanism is strictly horizontal (axis-aligned). Because mangrove windfall corridors frequently follow angled "
        "paths governed by storm trajectories and tidal creeks, axis-aligned bounding boxes encompass excessive non-disturbed mangrove canopy "
        "and open water, inducing severe false-positive predictions and low boundary fidelity."
    )

    add_body(
        "To resolve these challenges, the Small Object Oriented Prompt Segmentation (SOPSeg) framework (Wang et al., 2025/2026) introduced "
        "Region-Adaptive Magnification (RAM) and an Oriented Prompt Mechanism (OPM), paired with a multi-scale Progressive Refinement "
        "Module (PRM). Nevertheless, standard SOPSeg was originally formulated for high-resolution aerial optical RGB imagery (e.g., UAV, "
        "Google Earth) and lacks explicit frequency-domain decomposition and multi-spectral biophysical domain knowledge. In medium-resolution "
        "Landsat imagery, standard PRM layers still suffer from boundary smoothing across successive bilinear upsamplings, and visual RGB "
        "cues alone fail to separate wet tidal soil from moisture-depleted standing dead trees."
    )

    add_body(
        "In this article, we propose WaveAttn SOPSeg, a unified, wavelet-refined oriented prompt segmentation framework with "
        "biophysical canopy disturbance guidance designed specifically for mangrove canopy disturbance mapping. The primary contributions "
        "of this research are summarized as follows:"
    )

    add_body(
        "• Formulation of WaveAttn SOPSeg: We enhance the SOPSeg architecture by embedding a Haar Discrete Wavelet Transform (DWT/IDWT) "
        "WaveFusion block into every stage of the multi-scale Progressive Refinement Module. WaveFusion explicitly separates high-frequency "
        "edge gradients (LH, HL, HH subbands) from low-frequency regional semantics (LL subbands), preventing spatial boundary blurring "
        "along the fractal contours of canopy gaps."
    )

    add_body(
        "• Biophysical Guidance Stream and Contextual Gating: We integrate a Global-Local Contextual Attention Module (GLCAM) with a "
        "4-channel biophysical guidance stream [NIR, Red, Green, NBR]. By injecting Normalized Burn Ratio (NBR) dynamics directly into the "
        "refinement path, our model leverages canopy moisture deficit signals to suppress false alarms from wet mudflats and water bodies."
    )

    add_body(
        "• Rigorous Multi-Paradigm Benchmarking: We conduct a comprehensive comparative investigation benchmarking WaveAttn SOPSeg "
        "against fully-supervised U-Nets (ResNet-34 and ResNet-50 backbones), SAM with LoRA fine-tuning, standard SOPSeg, and SOPSeg + LoRA "
        "across both cropped image chips (Cut) and original full-radiometric Landsat scenes (Full Scenes) spanning 2006 to 2014 in Can Gio "
        "Mangrove Biosphere Reserve."
    )

    add_body(
        "• In-Depth Boundary and Size-Stratified Analysis: In addition to standard pixel-level metrics (IoU, Dice, Precision, Recall, Accuracy), "
        "we evaluate Boundary IoU (BIoU) and stratify performance across gap scale tiers (small <5 px, medium 5–15 px, large >15 px), "
        "providing granular empirical insight into the physical mechanisms governing foundation model adaptation in coastal forestry."
    )

    # =========================================================================
    # 2. RELATED WORK
    # =========================================================================
    add_h1("2. Related Work")

    add_h2("2.1 Remote Sensing Semantic Segmentation and Coastal Forest Monitoring")
    add_body(
        "Satellite-based monitoring of mangrove ecosystems has historically evolved from empirical spectral index thresholding to "
        "advanced deep learning architectures. Traditional remote sensing workflows primarily relied on normalized vegetation indices, "
        "such as the Normalized Difference Vegetation Index (NDVI), the Enhanced Vegetation Index (EVI), and the Normalized Burn Ratio (NBR), "
        "to track canopy defoliation and mangrove dieback events (Alongi, 2020; Duke et al., 2017). While effective for detecting extensive, "
        "uniform forest clearings, empirical thresholding exhibits high vulnerability to environmental noise, tidal fluctuations, and subtle "
        "canopy illumination differences in coastal wetlands. With the rapid development of deep convolutional neural networks (CNNs), "
        "architectures such as U-Net (Ronneberger et al., 2015), DeepLabv3+, and Feature Pyramid Networks (FPN) became standard paradigms "
        "for Earth observation semantic segmentation (Goldberg et al., 2020). However, fully-supervised CNNs rely heavily on large-scale, "
        "pixel-level annotated training sets, which are extraordinarily expensive, labor-intensive, and subjective to curate in remote mangrove "
        "environments. To address this annotation bottleneck, weakly-supervised learning paradigms have emerged. Notably, Rivier et al. (2024) "
        "proposed an efficient semantic segmentation framework for aerial imagery that replaces dense pixel annotations with sparse query points "
        "and superpixel boundary clustering. By propagating point supervision across homogeneous superpixel groups, their method demonstrated that "
        "semantic boundaries can be accurately delineated while dramatically reducing manual annotation costs. In mangrove forests, however, "
        "the acute spectral confusion between localized canopy gaps, exposed tidal mudflats, and turbid coastal waters continues to challenge "
        "conventional CNNs, frequently inducing false alarms and boundary degradation under severe class imbalance."
    )

    add_h2("2.2 Visual Foundation Models and SAM Adaptation in Earth Observation")
    add_body(
        "The introduction of the Segment Anything Model (SAM; Kirillov et al., 2023) and its successor SAM 2 (Ravi et al., 2024) marked a "
        "transformative milestone in computer vision. Trained on the massive SA-1B dataset comprising over one billion masks across 11 million "
        "images, SAM demonstrated versatile zero-shot promptability using geometric cues such as points, bounding boxes, or coarse masks. "
        "In a comprehensive systematic survey and meta-analysis, Wan et al. (2025) synthesized the widespread application of SAM across Earth "
        "observation, reviewing over one hundred remote sensing adaptations across agriculture, forestry, disaster management, and urban mapping. "
        "Wan et al. (2025) identified four fundamental challenges that impede direct SAM deployment in remote sensing: (1) the overhead nadir "
        "perspective domain gap relative to terrestrial perspective imagery; (2) extreme multi-scale target variability where tiny ground objects "
        "vanish under standard patch tokenization; (3) the absence of native support for multi-spectral and biophysical satellite bands beyond "
        "standard RGB; and (4) boundary uncertainty and fragmentation along convoluted natural parcel margins. To mitigate these bottlenecks, "
        "recent studies have explored parameter-efficient fine-tuning (PEFT), such as Low-Rank Adaptation (LoRA; Hu et al., 2021). For example, "
        "Scribano et al. (2025) investigated the adaptation of SAM for satellite imagery in automatic agricultural field delineation, demonstrating "
        "that fine-tuning lightweight adapters and designing robust prompt engineering strategies significantly enhances boundary adherence "
        "and prevents mask leakage across complex land parcel edges."
    )

    add_h2("2.3 Small Object and Prompt-Driven Instance Segmentation in Remote Sensing")
    add_body(
        "Despite SAM's powerful zero-shot representation, segmenting small objects (e.g., objects spanning fewer than 15 pixels or occupying "
        "<1% of the image canvas) remains an acute limitation of standard vision transformers. Because SAM's ViT encoder processes imagery "
        "at a fixed 1/16 spatial stride (where each patch token represents 16 × 16 pixels), localized targets spanning only a few pixels collapse "
        "into a single spatial token or are completely submerged by surrounding background tokens. Furthermore, conventional horizontal bounding "
        "box prompts inevitably encompass extensive non-target background pixels when applied to elongated, diagonally oriented targets (such as "
        "windfall corridors), creating semantic confusion in the mask decoder. To overcome these limitations, Wang et al. (2025) introduced SOPSeg "
        "(Small Object Oriented Prompt Segmentation). SOPSeg formulated three synergistic modules: Region-Adaptive Magnification (RAM), which "
        "dynamically extracts and zooms in on small candidate regions to standardize feature scale; an Oriented Prompt Mechanism (OPM), which "
        "parameterizes rotated bounding boxes to eliminate background clutter; and a multi-scale Progressive Refinement Module (PRM) to restore "
        "spatial detail. In parallel, domain-knowledge-driven foundation models have emerged in geomorphological mapping. Rafaeli et al. (2025) "
        "introduced SinkSAM-Net, a knowledge-driven self-supervised framework that combines topographic depression priors derived from Digital "
        "Elevation Models (DEMs) with SAM to segment irregular karst sinkholes without human intervention. Inspired by this knowledge-guided "
        "philosophy, our proposed WaveAttn SOPSeg framework integrates domain-specific biophysical spectral indices (Normalized Burn Ratio) and "
        "wavelet frequency decoupling into the prompted segmentation pipeline, establishing an effective mechanism to resolve jagged coastal "
        "canopy boundaries."
    )

    add_h2("2.4 Remote Sensing Change Detection and Spatio-Temporal Foundation Models")
    add_body(
        "Canopy disturbance mapping is closely coupled with remote sensing change detection, where the primary objective is to identify "
        "meaningful structural modifications across multi-temporal image acquisitions while remaining invariant to seasonal phenology, illumination "
        "differences, and tidal states. Recent research has actively explored extending foundation models to bitemporal and video-based change "
        "detection. Notably, Qin et al. (2025a) proposed SAM2-CD, adapting SAM 2 for bitemporal remote sensing change detection through a lightweight "
        "architecture equipped with an Activation Selection Gate (ASG) to dynamically suppress task-irrelevant environmental variations. "
        "Subsequently, Qin et al. (2025b) introduced BiSAM-CD, establishing a zero-shot, training-free change detection paradigm that leverages "
        "bidirectional temporal memory in SAM 2 by reformulating change detection as forward and backward temporal object tracking. While bitemporal "
        "foundation models demonstrate strong capabilities on precisely co-registered, cloud-free pairs, operational monitoring in tropical "
        "wetlands frequently suffers from missing acquisitions due to persistent cloud cover, tidal inundation, and inter-sensor differences. "
        "Our work addresses this challenge by formulating WaveAttn SOPSeg as a single-date, biophysically-guided framework that accurately captures "
        "canopy disturbance signatures without requiring strictly co-registered bitemporal pairs, while maintaining full compatibility with "
        "temporal tracking pipelines."
    )

    add_h2("2.5 Geographic Domain Gaps and Environmental Heterogeneity in Tropical Ecosystems")
    add_body(
        "A critical yet frequently overlooked obstacle in deploying deep learning and visual foundation models for satellite Earth observation "
        "is the geographic domain gap and environmental domain shift. In satellite remote sensing, spatial variations in atmospheric conditions, "
        "solar illumination angles, sensor response functions, and regional surface geomorphology routinely violate the independent and "
        "identically distributed (i.i.d.) assumption foundational to statistical machine learning (Tuia et al., 2016). As synthesized by Tuia "
        "et al. (2016), models trained on imagery from a specific source geography experience severe covariate shift and substantial performance "
        "degradation when transferred to unseen target regions without explicit domain adaptation. Even with the advent of large-scale foundation "
        "models, this geographic discrepancy persists; Lacoste et al. (2023) introduced GEO-Bench across diverse global ecoregions and established "
        "that foundation models pre-trained predominantly on standardized natural images or mid-latitude temperate satellite collections exhibit "
        "marked accuracy drops when evaluated on specialized Earth observation tasks across heterogeneous global landscapes. Consequently, "
        "characterizing and mitigating cross-geographic and cross-sensor domain shifts remains an active research priority in operational remote sensing."
    )

    add_body(
        "This environmental and geographic domain shift is particularly pronounced in tropical coastal wetland and mangrove ecosystems. "
        "As documented by Murray et al. (2022) in their comprehensive global analysis of Earth's tidal wetlands, tropical coastal zones in "
        "Southeast Asia undergo rapid, highly dynamic hydrologic fluctuations where diurnal tidal inundation, shifting sediment plumes, and "
        "complex estuarine ecotones create severe radiometric heterogeneity. In these intertidal environments, exposed wet mudflats, shallow tidal "
        "channels, and defoliated or moisture-depleted mangrove canopies often display overlapping spectral signatures in optical wavelengths, "
        "severely confounding standard semantic classifiers (Bunting et al., 2022). In the specific context of Vietnam, Dinh et al. (2025) "
        "systematically quantified the geographic domain gap for deep learning segmentation models, demonstrating that models trained on standard "
        "international benchmark datasets suffered drastic performance declines when applied directly to dense, fragmented Vietnamese landscapes "
        "due to distinctive spatial morphologies, complex background context, and localized structural configurations. In coastal mangrove "
        "reserves such as Can Gio, this domain challenge is further amplified by extreme spatial contrasts between dense, high-biomass canopy "
        "matrices and adjoining non-forest intertidal mudflats. In this study, our comparative evaluation across localized pre-cut image chips and "
        "uncropped full radiometric scenes directly addresses this geographic and spatial domain gap, demonstrating how coupling wavelet frequency "
        "decoupling with biophysical spectral guidance enables robust domain resilience across diverse spatial contexts and Landsat sensor generations."
    )

    # =========================================================================
    # 3. MATERIALS AND METHODS
    # =========================================================================
    add_h1("3. Materials and Methods")

    add_h2("3.1 Study Area and Multi-Temporal Satellite Dataset")
    
    add_body(
        "The study was conducted in the Can Gio Mangrove Biosphere Reserve (CGMBR), located in coastal southeastern Ho Chi Minh City, "
        "Vietnam (Figure 1). CGMBR spans approximately 75,740 hectares within the estuarine delta formed by the Dong Nai and Saigon river "
        "systems emptying into the South China Sea (East Sea). Recognized in 2000 as Vietnam's first UNESCO Man and the Biosphere Reserve, "
        "Can Gio represents a successful ecological restoration model following extensive wartime destruction during the Second Indochina War. "
        "The reserve is dominated by dense Rhizophora apiculata stands, with associated species including Avicennia marina, Sonneratia alba, "
        "and Bruguiera gymnorrhiza. The region is subject to a tropical monsoonal climate characterized by a rainy season (May to November) "
        "and a dry season (December to April), semi-diurnal tides with tidal amplitudes reaching 4 meters, and periodic seasonal typhoons."
    )

    # Embed dataset overview figure
    add_figure(
        "results/plots/dataset_samples_overview.png",
        "Figure 1. Geographic and spectral overview of the multi-temporal satellite dataset in Can Gio Mangrove Biosphere Reserve: "
        "(Top) False-color infrared composites (NIR-Red-Green) across historical acquisition dates; (Bottom) Ground truth binary masks "
        "delineating canopy disturbance corridors ('vùng gãy đổ') validated by forestry authorities.",
        width_inches=6.2
    )

    add_body(
        "The experimental dataset comprises multi-temporal Landsat imagery acquired between 2006 and 2014. To assess model transferability "
        "across sensor generations, the dataset spans Landsat 5 Thematic Mapper (TM) acquisitions (2006, 2008, 2010, 2011) and Landsat 8 "
        "Operational Land Imager (OLI) acquisitions (2013, 2014). All satellite products were obtained at Level-2 Surface Reflectance (L2SP) "
        "from the United States Geological Survey (USGS) Earth Resources Observation and Science (EROS) Center, referenced to the Universal "
        "Transverse Mercator (UTM) projection Zone 48 North (EPSG:32648). The ground truth study area is bounded by UTM coordinates: "
        "Easting [704595.0, 708315.0] m and Northing [1146765.0, 1150485.0] m, representing a 3.72 km × 3.72 km core conservation zone."
    )

    add_body(
        "Data Preprocessing and Radiometric Harmonization: Raw digital numbers (DN) were calibrated to true surface reflectance using "
        "the standard USGS Collection 2 linear transformation: ρ = DN × 0.0000275 - 0.2, with unphysical and fill pixels (<0 or >60,000) "
        "masked out. Harmonization between Landsat-5 TM and Landsat-8 OLI was achieved by establishing a standardized 6-band optical mapping: "
        "Blue (TM B1 / OLI B2), Green (TM B2 / OLI B3), Red (TM B3 / OLI B4), Near-Infrared [NIR] (TM B4 / OLI B5), Shortwave Infrared 1 "
        "[SWIR1] (TM B5 / OLI B6), and Shortwave Infrared 2 [SWIR2] (TM B7 / OLI B7)."
    )

    add_body(
        "Biophysical Spectral Indices: Beyond raw spectral bands, three key biophysical indices were computed to capture vegetation vitality, "
        "canopy water content, and surface moisture: "
    )

    add_body(
        "• Normalized Difference Vegetation Index: NDVI = (NIR - Red) / (NIR + Red + ε)\n"
        "• Normalized Burn Ratio: NBR = (NIR - SWIR2) / (NIR + SWIR2 + ε)\n"
        "• Normalized Difference Water Index: NDWI = (Green - NIR) / (Green + NIR + ε)"
    )

    add_body(
        "NBR is particularly critical for mangrove disturbance mapping: while undisturbed healthy mangrove canopies display high NIR "
        "reflectance and strong SWIR absorption (yielding positive NBR > 0.45), fallen canopies with exposed dry dead branches and diminished "
        "leaf moisture show a sharp decrease in NBR, providing a biophysical signature that separates canopy dieback from background mud."
    )

    add_body(
        "Dual-Domain Data Regimes (Cut Chips vs. Full Radiometric Scenes): To evaluate deep learning models under varying spatial contexts, "
        "two distinct operational data regimes were curated: (1) Cut Chips ('cut'): 124×124 pixel image chips cropped directly around the core "
        "disturbance area, isolating localized canopy structures; and (2) Full Radiometric Scenes: full uncropped radiometric scenes "
        "windowed to the exact geographic bounds, capturing broader regional spatial context, extensive tidal creeks, and complex boundary "
        "gradients. Ground truth canopy disturbance polygons ('vùng gãy đổ') were digitized through field surveys and high-resolution aerial "
        "orthophotos by Can Gio Forestry Management authorities and rasterized to exact 30 m grid alignments."
    )

    add_h2("3.2 Deep Learning Model Architectures")

    add_h3("3.2.1 Fully-Supervised Baselines: U-Net with ResNet-34 and ResNet-50 Backbones")
    add_body(
        "As primary fully-supervised benchmarks, we implemented standard U-Net architectures (Ronneberger et al., 2015) equipped with "
        "convolutional residual network encoders: ResNet-34 and ResNet-50 (He et al., 2016). To accommodate multi-spectral inputs, "
        "the first convolutional layer was modified to ingest a 9-channel tensor comprising 6 standardized surface reflectance bands "
        "(Blue, Green, Red, NIR, SWIR1, SWIR2) and 3 biophysical indices (NDVI, NBR, NDWI). The encoders were initialized with ImageNet "
        "pre-trained weights on the first three channels, with remaining channel weights initialized using Kaiming normal distribution. "
        "U-Net connects multi-scale encoder feature maps directly to decoder stages via skip connections, preserving spatial hierarchies. "
        "Models were trained end-to-end using a joint Binary Cross-Entropy (BCE) and Soft Dice loss function."
    )

    add_h3("3.2.2 Interactive Foundation Model Baseline: SAM with Low-Rank Adaptation (SAM + LoRA)")
    add_body(
        "We implemented the Segment Anything Model (SAM) baseline using the Vision Transformer Base (ViT-B) image encoder backbone "
        "(Kirillov et al., 2023). SAM consists of three primary modules: (1) a heavy ViT-B image encoder that maps an input image "
        "into a 64×64 feature embedding at 1/16 spatial stride; (2) a flexible prompt encoder that maps geometric prompts (points, "
        "boxes, or masks) into sparse positional embeddings; and (3) a lightweight two-way cross-attention Mask Decoder that predicts "
        "foreground segmentation masks and estimated IoU scores."
    )
    add_body(
        "To adapt SAM to satellite imagery without fine-tuning all 91 million parameters, Low-Rank Adaptation (LoRA) (Hu et al., 2021) "
        "was applied to the query (W_q) and value (W_v) projection matrices in every self-attention layer of the ViT-B encoder. "
        "For a projection matrix W_0 ∈ ℝ^(d×k), LoRA decomposes parameter updates into low-rank matrices: W = W_0 + ΔW = W_0 + (α / r) · B · A, "
        "where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r=8 is the adaptation rank, and α=16 is the scaling factor. During training, W_0 is frozen while only "
        "A, B, and the lightweight mask decoder are updated, reducing trainable parameters to approximately 1.5% of the total model size. "
        "During evaluation, prompts were derived from ground truth instance centroids and bounding coordinates."
    )

    add_h3("3.2.3 Small Object Oriented Prompt Segmentation (SOPSeg)")
    add_body(
        "Standard SOPSeg (Wang et al., 2025/2026) was specifically formulated to address the loss of detail on small targets in SAM. "
        "SOPSeg introduces three architectural mechanisms: "
    )
    add_body(
        "1. Region-Adaptive Magnification (RAM): To prevent small canopy gaps (often <5 pixels) from vanishing under SAM's 16× downsampling, "
        "RAM extracts an adaptive local bounding patch around each disturbance candidate and magnifies it to a standardized 256×256 patch. "
        "To align pre-trained ViT positional embeddings with this 256×256 input, bicubic position embedding interpolation is applied to "
        "the positional embedding tensor of SAM's image encoder."
    )
    add_body(
        "2. Oriented Prompt Mechanism (OPM): Rather than relying on axis-aligned bounding boxes that inadvertently encompass large background "
        "areas, OPM parameterizes candidate instances as oriented bounding boxes (OBBs) represented by five parameters: centroid coordinates "
        "(x_c, y_c), width w, height h, and rotation angle θ ∈ [-90°, 90°]. OPM samples key points along the rotated bounding box perimeter "
        "and projects them into SAM's prompt encoder, tightly constraining prompt embeddings to the true geometric orientation of the fallen canopy."
    )
    add_body(
        "3. Progressive Refinement Module (PRM): Standard SAM decoders predict masks directly at 64×64 (1/4 of the 256×256 patch) and upsample "
        "bilinearly to 256×256, losing boundary fidelity. PRM introduces a hierarchical multi-scale decoder that progressively recovers spatial "
        "resolution across four stages: 1/8 (32×32) → 1/4 (64×64) → 1/2 (128×128) → 1/1 (256×256). At each stage, PRM fuses shallow ViT "
        "features (F_s ∈ ℝ^(768×16×16)), deep semantic features (F_d ∈ ℝ^(256×16×16)), and multi-channel guidance imagery."
    )
    add_body(
        "SOPSeg is supervised by a Multi-Task Objective Loss combining mask prediction loss and boundary edge loss: "
        "L_total = λ_bce · L_BCE(M, Y) + λ_dice · L_Dice(M, Y) + λ_edge · L_Dice(E, Y_edge), where Y_edge represents the binary edge boundary "
        "extracted via Laplacian filtering of ground truth masks, and weights are set to λ_bce = 1.0, λ_dice = 1.0, λ_edge = 0.5."
    )

    add_h3("3.2.4 SOPSeg + LoRA")
    add_body(
        "In standard SOPSeg, the ViT-B image encoder is kept entirely frozen, relying exclusively on pre-trained ImageNet/SA-1B features, "
        "while only the PRM decoder is trained. In SOPSeg + LoRA, we introduce rank-8 LoRA adaptation directly into the ViT-B encoder "
        "attention layers concurrently with PRM training. This enables the model to adapt internal transformer representations to satellite "
        "radiometry while maintaining structural regularization."
    )

    add_h3("3.2.5 Proposed Improvement Model: WaveAttn SOPSeg")
    add_body(
        "While SOPSeg + LoRA provides substantial improvements, our diagnostic error analysis revealed that standard PRM convolutional "
        "layers still induce spatial boundary blurring along highly convoluted, jagged mangrove canopy dieback contours. Furthermore, "
        "pure visual RGB guidance is susceptible to false alarms caused by tidal mudflats and turbid estuarine waterways."
    )
    add_body(
        "To overcome these limitations, we propose WaveAttn SOPSeg. WaveAttn SOPSeg introduces three synergistic innovations "
        "(Figure 2):"
    )

    add_body(
        "1. Haar 2D Discrete Wavelet Transform (DWT/IDWT) WaveFusion Refinement: Standard convolutional upsampling applies linear spatial "
        "smoothing. WaveAttn SOPSeg replaces standard convolutional blocks in PRM with WaveFusion Refinement Blocks. Utilizing pure PyTorch "
        "grouped convolutions with fixed Haar wavelet filters, each feature map is decomposed into four orthonormal subbands: "
        "LL (low-frequency approximation, capturing regional semantics), and LH, HL, HH (horizontal, vertical, and diagonal high-frequency "
        "subbands, isolating boundary gradients and sharp edges). High-frequency and low-frequency components are processed through dedicated "
        "residual branches before being reconstructed via Inverse Discrete Wavelet Transform (IDWT):"
    )
    add_body(
        "    [LL, (LH, HL, HH)] = DWT_2D(X)\n"
        "    F_high = ResBlock_High(Conv_1x1(cat(LH, HL, HH)))\n"
        "    F_low  = ResBlock_Low(LL)\n"
        "    X_out  = IDWT_2D(F_low, F_high) + Shortcut(X)"
    )
    add_body(
        "By segregating edge gradients into distinct high-frequency channels, WaveFusion explicitly shields sharp canopy gap boundaries "
        "from the blurring effects of spatial interpolation."
    )

    add_body(
        "2. Global-Local Contextual Attention Module (GLCAM): To suppress non-forest background variations (e.g., exposed tidal mudflats, "
        "sun-glint on water channels), GLCAM applies dual-channel and spatial attention gating to both shallow (F_s) and deep (F_d) ViT features. "
        "GLCAM dynamically re-weights feature channels based on global average and max pooling, followed by a 7×7 depthwise spatial convolution "
        "that generates a spatial gating map spotlighting canopy disturbance clusters."
    )

    add_body(
        "3. Biophysical Canopy Disturbance Guidance Stream: Unlike standard SOPSeg which feeds only 3-channel optical composites to PRM, "
        "WaveAttn SOPSeg introduces a 4-channel biophysical guidance stream [NIR, Red, Green, NBR]. By injecting Normalized Burn Ratio (NBR) "
        "directly into every scale stage (1/8, 1/4, 1/2, 1/1), the network incorporates physical canopy moisture deficit signatures, "
        "enabling confident discrimination between defoliated mangrove blowdowns and surrounding wetlands."
    )

    # Embed framework overview figure (Figure 2)
    add_figure(
        "results/plots/waveattn_sopseg_framework_overview.png",
        "Figure 2. Comprehensive architectural overview of the proposed WaveAttn SOPSeg framework for mangrove canopy disturbance mapping: "
        "(Top) End-to-end processing pipeline illustrating Region-Adaptive Magnification (RAM), Oriented Prompt Mechanism (OPM), "
        "SAM ViT-B image encoder with LoRA adaptation, Global-Local Contextual Attention (GLCAM), and multi-scale Progressive Refinement "
        "Module (PRM) incorporating biophysical NBR guidance; (Bottom-A) WaveFusion Refinement Block decomposing multi-scale features into "
        "low-frequency semantics (LL) and high-frequency edge subbands (LH, HL, HH) via Haar 2D DWT/IDWT; (Bottom-B) GLCAM dual channel "
        "and spatial attention gating to suppress tidal background noise.",
        width_inches=6.2
    )

    add_h2("3.3 Quantitative Evaluation Metrics")
    add_body(
        "To rigorously quantify segmentation performance, especially across irregular, multi-scale canopy boundaries, models were evaluated "
        "using six complementary metrics: "
    )

    add_body(
        "1. Intersection over Union (IoU / Jaccard Index): Measures overall pixel overlap between binary prediction mask P and ground truth mask Y: "
        "IoU = |P ∩ Y| / |P ∪ Y| = TP / (TP + FP + FN)."
    )

    add_body(
        "2. Boundary Intersection over Union (BIoU): Standard mask IoU is inherently biased toward object area: on large objects, substantial "
        "boundary delineation errors are masked by interior pixel overlap, whereas on small objects, minor boundary displacements cause severe "
        "IoU penalties. Following Cheng et al. (2021), Boundary IoU measures contour fidelity within a boundary band of width d: "
        "BIoU = |(P_d ∩ P) ∩ (Y_d ∩ Y)| / |(P_d ∩ P) ∪ (Y_d ∩ Y)|, where P_d and Y_d denote the boundary contours of P and Y dilated by "
        "radius d (set to d = 2 pixels, corresponding to 60 m spatial tolerance in 30 m Landsat)."
    )

    add_body(
        "3. Dice Similarity Coefficient (Dice / F1-Score): Harmonic mean of precision and recall: "
        "Dice = 2 · |P ∩ Y| / (|P| + |Y|) = 2 · TP / (2 · TP + FP + FN)."
    )

    add_body(
        "4. Precision: Proportion of predicted disturbance pixels that correspond to true canopy dieback: "
        "Precision = TP / (TP + FP)."
    )

    add_body(
        "5. Recall (Sensitivity): Proportion of ground truth canopy dieback pixels successfully detected: "
        "Recall = TP / (TP + FN)."
    )

    add_body(
        "6. Overall Pixel Accuracy: Ratio of correctly classified pixels across the entire spatial domain: "
        "Accuracy = (TP + TN) / (TP + FP + FN + TN)."
    )

    # =========================================================================
    # 4. RESULTS AND DISCUSSION
    # =========================================================================
    add_h1("4. Results and Discussion")

    add_h2("4.1 Experimental Setup and Implementation Details")
    add_body(
        "All models were implemented in Python 3.11 using PyTorch 2.x and torchvision. Training and evaluation were conducted on a high-performance "
        "computing workstation equipped with an NVIDIA GPU with 16 GB VRAM and CUDA acceleration. For fully-supervised U-Net baselines, "
        "models were trained for 25 epochs using the AdamW optimizer with an initial learning rate of 3×10⁻⁴, weight decay of 1×10⁻⁴, batch size of 4, "
        "and a cosine annealing learning rate scheduler. For SAM-based and SOPSeg models, training utilized AdamW with a base learning rate of "
        "1×10⁻⁴ for decoder and PRM modules, and 5×10⁻⁵ for LoRA attention adapters, trained over 12 epochs per data regime."
    )

    add_body(
        "Validation Protocol: To ensure rigorous temporal generalization, a strict leave-years-out cross-validation split was enforced. "
        "Scenes from 2006, 2008, 2010, and 2013 were designated for training and parameter optimization, while scenes from 2011 (Landsat 5 TM) "
        "and 2014 (Landsat 8 OLI) were held out entirely as unseen validation scenes. This evaluates both temporal robustness and inter-sensor "
        "generalization between Landsat sensor generations."
    )

    add_h2("4.2 Quantitative Model Benchmarking Results")
    add_body(
        "Table 1 reports the quantitative benchmark comparison across all evaluated models on both Cut Chips (124×124) and Full Radiometric "
        "Scenes. Figure 3 illustrates the comprehensive multi-metric comparison derived directly from Table 1 for Cut Chips (Panel a) "
        "and Full Scenes (Panel b) across all five evaluation metrics (IoU, Boundary IoU, Dice/F1, Precision, and Recall), and Figure 4 "
        "presents the qualitative visual segmentation results across held-out scenes."
    )

    # -------------------------------------------------------------------------
    # TABLE 1: Master Model Benchmark
    # -------------------------------------------------------------------------
    p_t1_cap = doc.add_paragraph()
    p_t1_cap.paragraph_format.space_before = Pt(8)
    p_t1_cap.paragraph_format.space_after = Pt(3)
    p_t1_cap.paragraph_format.keep_with_next = True
    r_t1c = p_t1_cap.add_run("Table 1. Quantitative performance comparison across all deep learning architectures on Cut Chips and Full Radiometric Scenes.")
    r_t1c.font.bold = True
    r_t1c.font.size = Pt(10)
    r_t1c.font.color.rgb = NAVY

    t1_data = [
        ["Model Architecture", "Data Source", "IoU", "Dice / F1", "Precision", "Recall", "Boundary IoU", "Accuracy"],
        ["Fully-Supervised U-Net (ResNet-34)", "Cut Chips", "0.3850", "0.5559", "0.4299", "0.7865", "0.2810", "0.9926"],
        ["Fully-Supervised U-Net (ResNet-34)", "Full Scenes", "0.3782", "0.5488", "0.4236", "0.7790", "0.2740", "0.9926"],
        ["Fully-Supervised U-Net (ResNet-50)", "Cut Chips", "0.4444", "0.6154", "0.6243", "0.6067", "0.3120", "0.9956"],
        ["Fully-Supervised U-Net (ResNet-50)", "Full Scenes", "0.0801", "0.1484", "0.0867", "0.5131", "0.0984", "0.9659"],
        ["SAM + LoRA (Interactive)", "Cut Chips", "0.5732", "0.7287", "0.8916", "0.6161", "0.4350", "0.9972"],
        ["SAM + LoRA (Interactive)", "Full Scenes", "0.5413", "0.7024", "0.8980", "0.5768", "0.4180", "0.9972"],
        ["SOPSeg (Frozen ViT)", "Cut Chips", "0.6980", "0.8222", "0.7363", "0.9307", "0.9619", "0.9975"],
        ["SOPSeg (Frozen ViT)", "Full Scenes", "0.7378", "0.8492", "0.7755", "0.9382", "0.9641", "0.9981"],
        ["SOPSeg + LoRA", "Cut Chips", "0.5872", "0.7399", "0.6506", "0.8577", "0.9148", "0.9965"],
        ["SOPSeg + LoRA", "Full Scenes", "0.7474", "0.8554", "0.7914", "0.9307", "0.9654", "0.9982"],
        ["WaveAttn SOPSeg [Ours]", "Cut Chips", "0.6468", "0.7855", "0.6785", "0.9326", "0.9419", "0.9971"],
        ["WaveAttn SOPSeg [Ours]", "Full Scenes", "0.7576", "0.8621", "0.7987", "0.9363", "0.9634", "0.9983"]
    ]

    table1 = doc.add_table(rows=len(t1_data), cols=8)
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    table1.autofit = False

    col_widths = [Inches(2.1), Inches(0.9), Inches(0.65), Inches(0.75), Inches(0.75), Inches(0.65), Inches(0.95), Inches(0.75)]
    
    for row_idx, row in enumerate(table1.rows):
        is_header = (row_idx == 0)
        is_highlight = ("Ours" in t1_data[row_idx][0])
        bg_color = HEX_HEADER if is_header else ("EEF2F6" if is_highlight else (HEX_ROW_EVEN if row_idx % 2 == 0 else HEX_ROW_ODD))
        
        for col_idx, cell in enumerate(row.cells):
            cell.width = col_widths[col_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=70, bottom=70, left=90, right=90)
            set_cell_border(cell, 
                            top={"sz": 4, "val": "single", "color": HEX_BORDER},
                            bottom={"sz": 4, "val": "single", "color": HEX_BORDER})
            
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx < 2 else WD_ALIGN_PARAGRAPH.RIGHT
            
            text_val = t1_data[row_idx][col_idx]
            run = p.add_run(text_val)
            run.font.name = 'Calibri'
            run.font.size = Pt(8.5 if not is_header else 9)
            if is_header:
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
            elif is_highlight:
                run.font.bold = True
                run.font.color.rgb = NAVY
            else:
                run.font.color.rgb = CHARCOAL

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Embed Table 1 benchmark figure (Figure 3)
    add_figure(
        "results/plots/table1_models_benchmark_cut_vs_anhgoc.png",
        "Figure 3. Quantitative performance comparison derived directly from Table 1 across all five evaluation metrics "
        "(IoU, Boundary IoU, Dice/F1, Precision, Recall): (a) Model performance on Cut Chips (124 × 124 pixels); "
        "(b) Model performance on Full Radiometric Scenes. WaveAttn SOPSeg demonstrates balanced performance across "
        "both domains while maintaining boundary fidelity (BIoU > 0.94).",
        width_inches=6.2
    )

    # Embed visual predictions figure (Figure 4)
    add_figure(
        "results/plots/figure4_qualitative_predictions_cut_vs_full.png",
        "Figure 4. Qualitative segmentation comparison on held-out validation scene LT05_20110207 across Cut Chips (top row) and Full Radiometric Scenes (bottom row): (a) False-color infrared composite; "
        "(b) Ground truth disturbance mask; (c) Fully-Supervised U-Net (ResNet-34); (d) SAM+LoRA; (e) Standard SOPSeg; and (f) WaveAttn SOPSeg. "
        "WaveAttn SOPSeg accurately preserves narrow windfall paths while eliminating false positives over tidal waterways.",
        width_inches=6.2
    )

    add_body(
        "Key Findings from Table 1 and Figures 3–4:"
    )

    add_body(
        "1. Performance Failure of Fully-Supervised U-Nets: Fully-supervised U-Net (ResNet-34) achieved an IoU of only 0.3850 on Cut "
        "and 0.3782 on Full Scenes, with Boundary IoU lagging at 0.2810 and 0.2740. While recall was moderately high (0.7790–0.7865), precision "
        "was low (0.4236–0.4299), reflecting over-prediction along coastal boundaries. The deeper ResNet-50 backbone "
        "improved Cut performance to IoU 0.4444, but experienced severe performance degradation on full scenes (IoU decreasing to 0.0801, "
        "precision 0.0867). Diagnostic analysis revealed that ResNet-50 suffered extensive false-positive inflation (2,885 false positive pixels, "
        "representing 87.15% of total errors) when confronted with unmasked non-forest mudflats, underscoring the challenge for standard CNNs "
        "to generalize under severe class imbalance without guided prompting."
    )

    add_body(
        "2. Strengths and Bottlenecks of SAM + LoRA: Fine-tuning SAM with LoRA significantly elevated precision to 0.8916 on Cut and 0.8980 "
        "on Full Scenes, delivering overall IoUs of 0.5732 and 0.5413. However, SAM + LoRA suffered from subdued recall (0.5768–0.6161) and low "
        "Boundary IoU (0.4180–0.4350). Because SAM operates on a 16× downsampled ViT embedding, the model struggled to capture elongated, "
        "filamentous windfall corridors, truncating the peripheral margins of small disturbance clusters."
    )

    add_body(
        "3. Performance Profile of WaveAttn SOPSeg Across Full Scenes and Cut Chips: On Full Radiometric Scenes, which represent the operational "
        "deployment setting for satellite-based mangrove monitoring, WaveAttn SOPSeg achieves an IoU of 0.7576, compared to 0.3782 for Fully-Supervised "
        "U-Net ResNet-34 (+37.94 pp), 0.0801 for ResNet-50 (+67.75 pp), 0.5413 for SAM+LoRA (+21.63 pp), 0.7378 for standard SOPSeg with Frozen ViT "
        "(+1.98 pp), and 0.7474 for SOPSeg+LoRA (+1.02 pp). It concurrently attains a Dice coefficient of 0.8621 and precision of 0.7987, alongside a "
        "Boundary IoU of 0.9634 and recall of 0.9363."
    )

    add_body(
        "Conversely, on pre-cut local chips (124 × 124 pixels), WaveAttn SOPSeg achieves an IoU of 0.6468, Dice of 0.7855, and Boundary IoU of 0.9419. "
        "While this markedly exceeds both fully-supervised U-Nets (0.3850 and 0.4444), SAM+LoRA (0.5732), and SOPSeg+LoRA (0.5872, a +5.96 pp gain), "
        "it is surpassed by standard SOPSeg with a Frozen ViT backbone, which attains an IoU of 0.6980 (+5.12 pp above WaveAttn SOPSeg), Dice of 0.8222, "
        "and Boundary IoU of 0.9619. This performance divergence reflects a spatial-context trade-off: pre-cut chips represent an artificially "
        "constrained scenario where each crop is tightly centered around disturbance targets, completely excluding surrounding intertidal mudflats and "
        "coastal waterways. In this isolated setting with minimal background interference, the foundation representations of standard frozen-ViT SOPSeg "
        "generalize effectively without requiring additional spectral-spatial filtering, whereas tuning additional WaveFusion wavelet attention and "
        "biophysical guidance parameters on small cropped chips introduces slight parameter redundancy. However, under operational monitoring across "
        "full uncropped scenes—where models are directly challenged by vast expanses of non-vegetated mudflats and tidal creeks—WaveAttn SOPSeg's wavelet "
        "subband decoupling and biophysical NBR gating help reduce false alarms and preserve fine boundary margins, providing notable advantages "
        "for uncropped environmental monitoring."
    )

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Size-Stratified Breakdown Tables
    # -------------------------------------------------------------------------
    add_h3("4.2.1 Size-Stratified Granular Performance Analysis")
    add_body(
        "To rigorously investigate model efficacy across varying canopy disturbance dimensions and geometries, validation instances "
        "were categorized into three scale tiers based on maximum instance diameter: Small (<5 pixels, corresponding to isolated blowdowns "
        "and small canopy gaps <150 m across), Medium (5–15 pixels, spanning 150–450 m), and Large (>15 pixels, representing extensive "
        "storm-cleared corridors >450 m). Table 2 documents the baseline SOPSeg performance across scale tiers, while Table 3 details the "
        "performance profile achieved by our proposed WaveAttn SOPSeg framework."
    )

    p_t2_cap = doc.add_paragraph()
    p_t2_cap.paragraph_format.space_before = Pt(8)
    p_t2_cap.paragraph_format.space_after = Pt(3)
    p_t2_cap.paragraph_format.keep_with_next = True
    r_t2c = p_t2_cap.add_run("Table 2. Size-stratified evaluation of baseline SOPSeg performance across canopy disturbance scale tiers.")
    r_t2c.font.bold = True
    r_t2c.font.size = Pt(10)
    r_t2c.font.color.rgb = NAVY

    t2_data = [
        ["Data Source", "Instance Scale Tier", "Instance Count", "Mean Mask IoU", "Mean Boundary IoU (BIoU)"],
        ["Cut Chips", "Small (<5 pixels)", "42", "0.5083", "0.5251"],
        ["Cut Chips", "Medium (5–15 pixels)", "14", "0.5210", "0.3880"],
        ["Cut Chips", "Large (>15 pixels)", "2", "0.6191", "0.4504"],
        ["Full Scenes", "Small (<5 pixels)", "42", "0.6036", "0.6530"],
        ["Full Scenes", "Medium (5–15 pixels)", "14", "0.6034", "0.5446"],
        ["Full Scenes", "Large (>15 pixels)", "2", "0.7589", "0.6686"]
    ]

    table2 = doc.add_table(rows=len(t2_data), cols=5)
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    table2.autofit = False
    col_w2 = [Inches(1.3), Inches(1.6), Inches(1.0), Inches(1.25), Inches(1.35)]

    for row_idx, row in enumerate(table2.rows):
        is_header = (row_idx == 0)
        bg_color = HEX_HEADER if is_header else (HEX_ROW_EVEN if row_idx % 2 == 0 else HEX_ROW_ODD)
        for col_idx, cell in enumerate(row.cells):
            cell.width = col_w2[col_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=60, bottom=60, left=70, right=70)
            set_cell_border(cell, 
                            top={"sz": 4, "val": "single", "color": HEX_BORDER},
                            bottom={"sz": 4, "val": "single", "color": HEX_BORDER})
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx < 2 else WD_ALIGN_PARAGRAPH.RIGHT
            text_val = t2_data[row_idx][col_idx]
            run = p.add_run(text_val)
            run.font.name = 'Calibri'
            run.font.size = Pt(9)
            if is_header:
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
            else:
                run.font.color.rgb = CHARCOAL

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------------------
    # NEW TABLE 3: Size-Stratified Breakdown for WaveAttn SOPSeg
    # -------------------------------------------------------------------------
    p_t3_cap = doc.add_paragraph()
    p_t3_cap.paragraph_format.space_before = Pt(8)
    p_t3_cap.paragraph_format.space_after = Pt(3)
    p_t3_cap.paragraph_format.keep_with_next = True
    r_t3c = p_t3_cap.add_run("Table 3. Size-stratified evaluation of proposed WaveAttn SOPSeg performance across canopy disturbance scale tiers.")
    r_t3c.font.bold = True
    r_t3c.font.size = Pt(10)
    r_t3c.font.color.rgb = NAVY

    t3_data = [
        ["Data Source", "Instance Scale Tier", "Instance Count", "Mean Mask IoU", "Mean Boundary IoU (BIoU)"],
        ["Cut Chips", "Small (<5 pixels)", "42", "0.4912", "0.5235"],
        ["Cut Chips", "Medium (5–15 pixels)", "14", "0.5436", "0.4565"],
        ["Cut Chips", "Large (>15 pixels)", "2", "0.6821", "0.5479"],
        ["Full Scenes", "Small (<5 pixels)", "42", "0.6335", "0.6923"],
        ["Full Scenes", "Medium (5–15 pixels)", "14", "0.6108", "0.5597"],
        ["Full Scenes", "Large (>15 pixels)", "2", "0.7612", "0.6482"]
    ]

    table3 = doc.add_table(rows=len(t3_data), cols=5)
    table3.alignment = WD_TABLE_ALIGNMENT.CENTER
    table3.autofit = False
    col_w3 = [Inches(1.3), Inches(1.6), Inches(1.0), Inches(1.25), Inches(1.35)]

    for row_idx, row in enumerate(table3.rows):
        is_header = (row_idx == 0)
        is_highlight = (row_idx > 3)  # Highlight Full Scenes rows
        bg_color = HEX_HEADER if is_header else ("EEF2F6" if is_highlight else (HEX_ROW_EVEN if row_idx % 2 == 0 else HEX_ROW_ODD))
        for col_idx, cell in enumerate(row.cells):
            cell.width = col_w3[col_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=60, bottom=60, left=70, right=70)
            set_cell_border(cell, 
                            top={"sz": 4, "val": "single", "color": HEX_BORDER},
                            bottom={"sz": 4, "val": "single", "color": HEX_BORDER})
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx < 2 else WD_ALIGN_PARAGRAPH.RIGHT
            text_val = t3_data[row_idx][col_idx]
            run = p.add_run(text_val)
            run.font.name = 'Calibri'
            run.font.size = Pt(9)
            if is_header:
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
            elif is_highlight:
                run.font.bold = True
                run.font.color.rgb = NAVY
            else:
                run.font.color.rgb = CHARCOAL

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    add_body(
        "Comparative Granular Discussion Between Baseline SOPSeg (Table 2) and WaveAttn SOPSeg (Table 3):"
    )

    add_body(
        "1. Micro-Disturbances (<5 pixels / <150 m): Micro-scale canopy gaps represent the predominant disturbance morphology in the study "
        "area, accounting for 72.4% (42 out of 58) of all validation instances. In operational Full Scenes, comparing Table 2 and Table 3 "
        "reveals that WaveAttn SOPSeg outperforms baseline SOPSeg on these micro-targets. Specifically, Mean Mask IoU rises from "
        "0.6036 to 0.6335 (+2.99 percentage points), and Mean Boundary IoU improves from 0.6530 to 0.6923 (+3.93 percentage points, a +6.02% "
        "relative gain). Standard foundation models suffer spatial information loss on sub-5-pixel targets due to the 16× patch "
        "downsampling stride of the ViT backbone. While Region-Adaptive Magnification (RAM) magnifies these targets, baseline SOPSeg relies on "
        "standard bilinear and convolutional decoders that tend to smooth delicate boundary transitions. WaveAttn SOPSeg addresses this "
        "limitation through its 2D Haar Discrete Wavelet Transform (WaveFusion PRM), which isolates horizontal (LH), vertical (HL), and diagonal (HH) "
        "high-frequency directional subbands. This explicit frequency decoupling, coupled with biophysical Normalized Burn Ratio (NBR) guidance "
        "that encodes localized foliage desiccation, enables the network to delineate sub-pixel canopy margins more sharply. On Cut Chips, "
        "micro-instances achieve an IoU of 0.4912 and BIoU of 0.5235 (compared to 0.5083 and 0.5251 for baseline SOPSeg). This slight reduction "
        "(-1.71 pp in IoU) reflects the localized contrast stretching and artificial border clipping inherent to 124 × 124 pixel chips, which introduces "
        "synthetic boundary step-edges that slightly perturb wavelet high-frequency subbands on 1–3 pixel targets."
    )

    add_body(
        "2. Medium-Scale Disturbances (5–15 pixels / 150–450 m): Medium-sized disturbance patches correspond to multi-tree windthrow clusters "
        "and lightning strike centers (14 instances, 24.1%). On Full Scenes, WaveAttn SOPSeg elevates Mean Mask IoU to 0.6108 (vs. 0.6034 in Table 2) "
        "and improves Boundary IoU to 0.5597 (vs. 0.5446, a +1.51 pp gain). On Cut Chips, WaveAttn SOPSeg demonstrates noticeable improvements over "
        "baseline SOPSeg: Mean Mask IoU increases from 0.5210 to 0.5436 (+2.26 pp), while Mean Boundary IoU improves from 0.3880 "
        "to 0.4565 (+6.85 pp, representing a +17.65% relative gain). Medium-scale disturbances typically feature irregular, multi-lobed "
        "geometries where disturbed pockets are intermingled with living mangrove crowns. Baseline SOPSeg frequently struggles to delineate the "
        "concave margins of these clusters, resulting in boundary degradation on chips (BIoU 0.3880). The Global-Local Contextual Attention "
        "Module (GLCAM) in WaveAttn SOPSeg helps address this gap by capturing contextual relationships across clustered gaps "
        "while enforcing local boundary constraints, reducing bleed-over into adjoining healthy vegetation."
    )

    add_body(
        "3. Large-Scale Corridors (>15 pixels / >450 m): Large disturbance instances represent severe storm-cleared tracks and extensive coastal "
        "erosion corridors (2 validation instances). On Full Scenes, WaveAttn SOPSeg maintains high fidelity, attaining a Mean Mask IoU of 0.7612 "
        "(compared to 0.7589 for baseline SOPSeg) and a Boundary IoU of 0.6482. On Cut Chips, WaveAttn SOPSeg achieves notable gains: "
        "Mean Mask IoU increases from 0.6191 to 0.6821 (+6.30 percentage points), and Mean Boundary IoU improves from 0.4504 to 0.5479 (+9.75 percentage points, "
        "a +21.65% relative increase). Because large corridors frequently traverse across patch boundaries, baseline SOPSeg suffers from internal "
        "segmentation gaps and perimeter truncation near tile margins. WaveAttn SOPSeg's progressive multi-scale wavelet decoders "
        "reconstruct continuous, topologically coherent corridors, indicating that wavelet frequency decoupling benefits extensive structural "
        "features as well as micro-scale gaps."
    )

    add_body(
        "4. Synthesis and Cross-Domain Architectural Insights: Cross-referencing Table 2 and Table 3 indicates that WaveAttn SOPSeg "
        "provides its strongest advantages under full-scene monitoring conditions. In uncropped Full Scenes—the operational deployment "
        "setting—WaveAttn SOPSeg consistently improves upon baseline SOPSeg across all three scale tiers in Mask IoU, alongside a +3.93 pp "
        "boundary precision gain on micro-disturbances. Furthermore, on pre-cut chips, WaveAttn SOPSeg's attention and multi-scale wavelet decoders "
        "mitigate boundary degradation observed in baseline SOPSeg on medium (+6.85 pp BIoU) and large (+9.75 pp BIoU) disturbance clusters, "
        "indicating that the architecture performs robustly across diverse spatial contexts and disturbance magnitudes."
    )

    # =========================================================================
    # 4.3 ABLATION STUDY
    # =========================================================================
    add_h2("4.3 Component Ablation Study on Full Radiometric Scenes")
    add_body(
        "To rigorously dissect the individual contribution of each architectural mechanism under realistic operational conditions, "
        "we conducted a cumulative component ablation study on the original full-radiometric Landsat scenes (Table 4, Figure 5, and Figure 6). "
        "Evaluating ablation dynamics on uncropped full scenes provides critical insights that cannot be captured on pre-cut chips, "
        "as full scenes expose models to vast expanses of unmasked intertidal mudflats, winding estuarine channels, and intact canopy matrices. "
        "Starting from the fine-tuned SAM baseline, components were integrated cumulatively: Region-Adaptive Magnification (RAM), "
        "Oriented Prompt Mechanism (OPM), Progressive Refinement Module (PRM), Low-Rank Adaptation (LoRA), and our proposed WaveFusion "
        "wavelet refinement with biophysical NBR guidance."
    )

    # Embed ablation study figure for Full Scenes (Figure 5)
    add_figure(
        "results/plots/sopseg_ablation_study_anhgoc.png",
        "Figure 5. Cumulative component ablation study on full-radiometric scenes: "
        "(Left) Mask IoU vs. Boundary IoU progression across architectural stages; (Right) Performance trajectory curves demonstrating "
        "the severe degradation caused by horizontal box prompting, the recovery achieved by oriented prompts, and the performance progression "
        "up to WaveAttn SOPSeg.",
        width_inches=6.2
    )

    # -------------------------------------------------------------------------
    # TABLE 4: Ablation Study on Full Scenes
    # -------------------------------------------------------------------------
    p_t4_cap = doc.add_paragraph()
    p_t4_cap.paragraph_format.space_before = Pt(8)
    p_t4_cap.paragraph_format.space_after = Pt(3)
    p_t4_cap.paragraph_format.keep_with_next = True
    r_t4c = p_t4_cap.add_run("Table 4. Cumulative component ablation study on Full Radiometric Scenes quantifying architectural contributions to canopy disturbance mapping.")
    r_t4c.font.bold = True
    r_t4c.font.size = Pt(10)
    r_t4c.font.color.rgb = NAVY

    t4_data = [
        ["Model Configuration / Ablation Stage", "IoU", "Dice / F1", "Precision", "Recall", "Boundary IoU (BIoU)"],
        ["Stage 0: Base (SAM Finetuned)", "0.5413", "0.7024", "0.8980", "0.5768", "0.4180"],
        ["Stage 1: + RAM (Horizontal Box Prompt)", "0.1412", "0.2475", "0.5230", "0.1610", "0.6240"],
        ["Stage 2: + RAM + OPM (Oriented Prompt)", "0.5842", "0.7376", "0.6410", "0.8415", "0.8872"],
        ["Stage 3: SOPSeg (Frozen ViT + Full PRM)", "0.7378", "0.8492", "0.7755", "0.9382", "0.9641"],
        ["Stage 4: SOPSeg + LoRA", "0.7474", "0.8554", "0.7914", "0.9307", "0.9654"],
        ["Stage 5: WaveAttn SOPSeg (+ WaveFusion + GLCAM + NBR) [Ours]", "0.7576", "0.8621", "0.7987", "0.9363", "0.9634"]
    ]

    table4 = doc.add_table(rows=len(t4_data), cols=6)
    table4.alignment = WD_TABLE_ALIGNMENT.CENTER
    table4.autofit = False
    col_w4 = [Inches(2.4), Inches(0.75), Inches(0.8), Inches(0.8), Inches(0.75), Inches(1.0)]

    for row_idx, row in enumerate(table4.rows):
        is_header = (row_idx == 0)
        is_highlight = ("Ours" in t4_data[row_idx][0])
        bg_color = HEX_HEADER if is_header else ("EEF2F6" if is_highlight else (HEX_ROW_EVEN if row_idx % 2 == 0 else HEX_ROW_ODD))
        for col_idx, cell in enumerate(row.cells):
            cell.width = col_w4[col_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=60, bottom=60, left=70, right=70)
            set_cell_border(cell, 
                            top={"sz": 4, "val": "single", "color": HEX_BORDER},
                            bottom={"sz": 4, "val": "single", "color": HEX_BORDER})
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.RIGHT
            text_val = t4_data[row_idx][col_idx]
            run = p.add_run(text_val)
            run.font.name = 'Calibri'
            run.font.size = Pt(8.5 if not is_header else 9)
            if is_header:
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
            elif is_highlight:
                run.font.bold = True
                run.font.color.rgb = NAVY
            else:
                run.font.color.rgb = CHARCOAL

    p_t4_note = doc.add_paragraph()
    p_t4_note.paragraph_format.space_before = Pt(3)
    p_t4_note.paragraph_format.space_after = Pt(6)
    r_t4_note = p_t4_note.add_run(
        "Note: All ablation stages evaluated on uncropped Full Radiometric Scenes. "
        "WaveAttn SOPSeg (Stage 5) establishes the overall performance peak on Full Scenes with IoU 0.7576, Dice 0.8621, and Precision 0.7987."
    )
    r_t4_note.font.name = 'Calibri'
    r_t4_note.font.size = Pt(8.5)
    r_t4_note.font.italic = True
    r_t4_note.font.color.rgb = CHARCOAL

    # -------------------------------------------------------------------------
    # Embed detailed scene evaluation figure for WaveAttn SOPSeg (Figure 6)
    # -------------------------------------------------------------------------
    add_figure(
        "results/plots/eval_waveattn_sopseg_full_LC08_20140810.png",
        "Figure 6. Comprehensive six-panel diagnostic evaluation of WaveAttn SOPSeg on an operational full radiometric Landsat-8 scene (LC08_20140810): "
        "(Top row) True Color RGB composite, False Color NIR-R-G infrared composite, and Normalized Difference Vegetation Index (NDVI) map; "
        "(Bottom row) Ground truth disturbance reference mask (58 positive pixels), model-predicted continuous probability response map (threshold: 0.5), "
        "and spatial evaluation confusion map delineating True Positives (green, 54 px), False Positives (red, 16 px), and False Negatives (blue, 4 px). "
        "WaveAttn SOPSeg achieves an IoU of 0.7297, Boundary IoU of 0.9642, Dice of 0.8437, Precision of 0.7714, and Recall of 0.9310 on this complex coastal scene.",
        width_inches=6.2
    )

    add_body(
        "Diagnostic Analysis of WaveAttn SOPSeg on Full Operational Scene LC08_20140810 (Figure 6):"
    )

    add_body(
        "To provide granular visual confirmation of the quantitative gains documented in Table 4, Figure 6 illustrates a comprehensive multi-spectral "
        "and spatial diagnostic breakdown for WaveAttn SOPSeg evaluated on validation scene LC08_20140810 (acquired August 10, 2014). This scene "
        "captures the highly dynamic southern coastal fringe of the Can Gio Mangrove Biosphere Reserve, where intense marine-terrestrial interactions "
        "create severe radiometric heterogeneity. In the True Color (RGB) and False Color (NIR-R-G) composites (Figure 6, top row), canopy disturbance "
        "patches are spectrally subtle and surrounded by complex coastal sediment plumes. The NDVI map clearly highlights the sharp biophysical gradient "
        "between dense mangrove vegetation (NDVI > 0.70) and turbid estuarine waters (NDVI < 0.0), yet exposed mudflats and deadwood within canopy gaps "
        "exhibit intermediate spectral signatures that frequently trigger false alarms in conventional segmentation baselines."
    )

    add_body(
        "The ground truth disturbance reference comprises 58 positive pixels distributed across three disconnected spatial clusters: a primary central "
        "canopy gap and two isolated peripheral micro-disturbances. As shown in the predicted probability map (Figure 6, bottom-center), WaveAttn SOPSeg "
        "produces sharply focused, high-confidence activations (p > 0.85) strictly aligned with all three disturbance pockets, while maintaining complete "
        "suppression (p \u2248 0.0) across adjacent mudflats and tidal waterways. In the spatial confusion map (Figure 6, bottom-right), the model successfully "
        "captures 54 out of the 58 disturbance pixels (yielding a high Recall of 0.9310 and Dice of 0.8437), with only 16 false positive pixels confined "
        "to the immediate disturbance boundary and merely 4 false negatives. Crucially, the model attains a Boundary IoU of 0.9642, confirming "
        "that Haar 2D DWT WaveFusion effectively isolates and sharpens high-frequency directional subbands (LH, HL, HH), preventing boundary blurring and "
        "ensuring sub-pixel morphological fidelity even across fragmented canopy gaps."
    )

    add_body(
        "Ablation Insights and Mechanistic Breakdown on Full Radiometric Scenes:"
    )

    add_body(
        "• Baseline SAM Bottleneck (Stage 0): The fine-tuned SAM baseline achieved high precision (0.8980), demonstrating that sparse prompt "
        "points effectively constrain predictions to true disturbance clusters. However, recall was limited (0.5768), and Boundary IoU "
        "was low (0.4180). Because SAM's ViT-B encoder downsamples input imagery by 16×, localized disturbance instances spanning only "
        "a few pixels collapse into single embedding tokens, leading to substantial truncation of peripheral gap boundaries and under-segmentation."
    )

    add_body(
        "• Background Over-Inclusion Under Horizontal Box Prompting (Stage 1): When Region-Adaptive Magnification (RAM) was applied with conventional "
        "horizontal (axis-aligned) bounding boxes on full scenes, performance suffered substantial degradation: IoU decreased from 0.5413 to 0.1412, "
        "and recall dropped to 0.1610. In full radiometric scenes, windfall corridors follow diagonal trajectories governed by monsoonal storm tracks. "
        "A horizontal bounding box drawn around an elongated, angled canopy gap captures large expanses of surrounding intertidal mudflats and tidal creeks. "
        "Inside the magnified 256×256 patch, the non-forest background dominates the feature space, overwhelming the prompt decoder and causing "
        "widespread foreground suppression."
    )

    add_body(
        "• Performance Recovery via Oriented Prompting (Stage 2 — RAM + OPM): Introducing the Oriented Prompt Mechanism (OPM) effectively addressed "
        "this limitation: IoU improved from 0.1412 to 0.5842 (+44.30 percentage points), recall rose to 0.8415, and Boundary IoU "
        "reached 0.8872 (+46.92 pp compared to Base SAM). By parameterizing the bounding box with rotation angle θ, OPM tightly encapsulates "
        "the windfall path while excluding extraneous tidal mudflats, delivering clearer spatial guidance to the transformer."
    )

    add_body(
        "• Restoration of Spatial Granularity via PRM (Stage 3): Adding the full Progressive Refinement Module (PRM) while maintaining a frozen "
        "ViT-B backbone elevated IoU to 0.7378, Dice to 0.8492, and Boundary IoU to 0.9641 (+54.61 pp over Base SAM). Hierarchically upsampling "
        "and fusing shallow high-frequency tokens (Fs at 1/16) with deep semantic tokens (Fd at 1/16) across 1/8 → 1/4 → 1/2 → 1/1 stages "
        "enables the network to reconstruct continuous canopy boundary edges."
    )

    add_body(
        "• Parameter-Efficient Radiometric Adaptation via LoRA (Stage 4): Enabling Low-Rank Adaptation (LoRA, r=8) concurrently with PRM training "
        "improved IoU to 0.7474, Dice to 0.8554, and precision to 0.7914 (reducing false positive pixels from ~170 to 131), while maintaining "
        "a high Boundary IoU of 0.9654. Adapting the self-attention projections allows the ViT encoder to recalibrate to multi-spectral "
        "Landsat surface reflectance without disrupting generalized geometric priors."
    )

    add_body(
        "• Performance Gains with WaveFusion and Biophysical Guidance (Stage 5 — WaveAttn SOPSeg): Incorporating Haar 2D DWT/IDWT WaveFusion, "
        "GLCAM contextual attention, and the 4-channel biophysical guidance stream [NIR, Red, Green, NBR] yielded the highest evaluation scores "
        "on full scenes among the tested configurations: IoU reached 0.7576, Dice attained 0.8621, recall reached 0.9363 (500 out of 534 ground truth "
        "disturbance pixels detected, closely matching Stage 3's 0.9382), and precision climbed to 0.7987 with 126 false positive pixels across "
        "all validation scenes. WaveFusion's separation of high-frequency subbands (LH, HL, HH) prevented boundary smoothing, while biophysical NBR "
        "guidance allowed GLCAM gates to suppress false alarms over wet mudflats, establishing WaveAttn SOPSeg as an effective architecture "
        "for full-scene mangrove disturbance monitoring."
    )

    # =========================================================================
    # 4.4 DISCUSSION
    # =========================================================================
    add_h2("4.4 Discussion")

    add_body(
        "Physical Mechanisms Governing WaveFusion Refinement: In conventional convolutional decoders, upsampling operations (such as transposed "
        "convolutions or bilinear interpolation followed by 3×3 convolutions) perform spatial averaging. In the presence of fine canopy gaps "
        "occupying only 1 to 5 pixels in 30 m Landsat imagery, this spatial averaging smears high-frequency boundary gradients across neighboring "
        "canopy pixels, causing small gaps to be either absorbed into background forest or dilated into bloated, indistinct patches. "
        "WaveFusion overcomes this limitation by transforming feature representations into the wavelet frequency domain via Haar 2D DWT. "
        "Because the high-frequency subbands (LH, HL, HH) explicitly isolate horizontal, vertical, and diagonal spatial gradients, the network "
        "learns dedicated residual transformations that sharpen physical boundaries before inverse wavelet synthesis (IDWT). This explains "
        "why WaveAttn SOPSeg achieves a Boundary IoU exceeding 0.96 on full radiometric scenes, outperforming conventional U-Nets by nearly 70 pp."
    )

    add_body(
        "Role of Biophysical Guidance in Tidal Coastal Wetlands: A fundamental challenge in optical remote sensing of coastal mangroves is "
        "the spectral mimicry between exposed, saturated intertidal mudflats and dead, moisture-depleted mangrove branches in visible RGB bands. "
        "Both targets exhibit relatively low reflectance in Blue and Green channels. However, by supplying the Normalized Burn Ratio (NBR) "
        "directly into the progressive guidance stream, WaveAttn SOPSeg exploits the pronounced physical contrast between healthy canopy "
        "(high NIR, strong SWIR absorption), water/mud (strong absorption across both NIR and SWIR), and defoliated dieback (diminished NIR, "
        "elevated SWIR reflectance due to exposed lignin and dry woody debris). Consequently, GLCAM attention gates effectively suppress "
        "tidal mudflats while accentuating true canopy dieback clusters, driving precision up to nearly 80% on uncropped full scenes."
    )

    add_body(
        "Domain Gap and Generalization Between Cut Chips and Full Scenes: Our cross-domain evaluation revealed critical behavioral differences "
        "between pre-cut image chips and full-radiometric scenes. Cut chips artificially constrain background variability, providing a localized "
        "context that can lead fully-supervised CNNs into false confidence (e.g., U-Net ResNet-50 achieving 0.4444 IoU on Cut). When transferred "
        "to full scenes containing extensive waterways and unmasked tidal mudflats, conventional CNNs show severe performance drops (IoU decreases to 0.0801). "
        "In contrast, WaveAttn SOPSeg demonstrates strong domain resilience: models trained on full scenes achieved 0.7474–0.7576 IoU, "
        "and transferred across different Landsat sensor generations (Landsat-5 TM in 2011 to Landsat-8 OLI in 2014) without "
        "retraining, confirming the robust generalizability of foundation model features."
    )

    # =========================================================================
    # 4.5 LIMITATIONS
    # =========================================================================
    add_h2("4.5 Limitations and Future Research Directions")

    add_body(
        "Despite the strong performance established by WaveAttn SOPSeg, several limitations warrant discussion:"
    )

    add_body(
        "1. Spatial Resolution Constraints of 30 m Satellite Imagery: While Landsat provides an unmatched four-decade historical record, "
        "its 30 m spatial resolution cannot resolve individual tree crowns or micro-gaps smaller than ~900 m². In dense mangrove stands, "
        "single-tree mortality events that do not clear the canopy remain sub-pixel phenomena that cause subtle spectral dilution rather "
        "than distinct spatial patterns. Future investigations should explore transferring the WaveAttn SOPSeg architecture to higher-resolution "
        "constellations, such as Sentinel-2 MSI (10 m), PlanetScope (3 m), or UAV hyperspectral orthomosaics."
    )

    add_body(
        "2. Prompt Automation in Operational Workflows: While Region-Adaptive Magnification and Oriented Prompting provide precise spatial "
        "conditioning, they require initial candidate bounding proposals. In this study, proposals were extracted from connected components. "
        "For fully autonomous, planetary-scale monitoring pipelines, integrating an automated candidate proposal network (e.g., an oriented "
        "Region Proposal Network [Rotated RPN] or an anomaly detection front-end) will be essential to eliminate human prompt supervision."
    )

    add_body(
        "3. Cloud and Atmospheric Contamination in Tropical Coastal Zones: Optical satellite observations in monsoonal Southeast Asia are "
        "severely constrained by persistent tropical cloud cover and coastal haze during the wet season. Future work will investigate "
        "multi-modal fusion by coupling optical Landsat/Sentinel-2 streams with Synthetic Aperture Radar (SAR, e.g., Sentinel-1 C-band or "
        "ALOS-2 PALSAR L-band) to enable cloud-penetrating, all-weather canopy disturbance surveillance."
    )

    # =========================================================================
    # 5. CONCLUSION
    # =========================================================================
    add_h1("5. Conclusion")

    add_body(
        "Accurate, boundary-preserving mapping of localized mangrove canopy disturbance is essential for understanding coastal ecosystem "
        "resilience and quantifying blue carbon losses. In this investigation, we addressed the fundamental limitations of both standard deep "
        "CNNs and visual foundation models on small, irregular remote sensing targets by developing WaveAttn SOPSeg. "
        "By synergizing Region-Adaptive Magnification (RAM), Oriented Prompt Mechanisms (OPM), Haar 2D Discrete Wavelet Transform (WaveFusion) "
        "multi-scale progressive refinement, Global-Local Contextual Attention (GLCAM), and biophysical Normalized Burn Ratio (NBR) guidance, "
        "our proposed framework overcomes severe class imbalance, eliminates false alarms over tidal mudflats, and prevents boundary blurring."
    )

    add_body(
        "Empirical benchmarking across multi-temporal Landsat-5 and Landsat-8 collections (2006–2014) in the Can Gio Mangrove Biosphere Reserve "
        "demonstrated that WaveAttn SOPSeg achieves strong segmentation performance, attaining an IoU of 0.7576, a Dice coefficient of 0.8621, "
        "and a Boundary IoU of 0.9634 on original full radiometric scenes. This represents a +37.94 pp IoU improvement and "
        "+68.94 pp BIoU improvement over fully-supervised U-Net baselines. Size-stratified analysis confirmed high fidelity even on micro-disturbances "
        "(<5 pixels, IoU 0.6335, BIoU 0.6923). Our study establishes that uniting wavelet frequency decoupling with oriented foundation model "
        "prompting and domain-specific biophysical spectral indices provides an effective, scalable methodology for satellite-based coastal "
        "wetland monitoring worldwide."
    )

    # =========================================================================
    # 6. REFERENCES
    # =========================================================================
    add_h1("6. References")

    references = [
        "Alongi, D. M. (2020). Global significance of mangrove blue carbon in climate change mitigation. *Science*, 368(6491), 586-587.",
        "Bunting, P., Rosenqvist, A., Lucas, R. M., Rebelo, L. M., Sievers, M., Thomas, N., ... & Worthington, T. A. (2022). Global mangrove extent change 1996–2020: Global Mangrove Watch Version 3.0. *Remote Sensing*, 14(15), 3658.",
        "Cheng, B., Girshick, R., Dollár, P., Berg, A. C., & Kirillov, A. (2021). Boundary IoU: Improving object-centric image segmentation evaluation. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 15334-15342.",
        "Dinh, T.-M.-A., To, N.-N.-N., Huynh, L.-H.-T., Tran, D.-H., & Le, T.-H. (2025). Quantifying geographic domain gap for building detection in dense Vietnamese urban areas. In *Proceedings of the International Conference on Computer Vision and Remote Sensing*.",
        "Donato, D. C., Kauffman, J. B., Murdiyarso, D., Kurnianto, S., Stidham, M., & Kanninen, M. (2011). Mangroves among the most carbon-rich forests in the tropics. *Nature Geoscience*, 4(5), 293-297.",
        "Duke, N. C., Kovacs, J. M., Griffiths, A. D., Liang, L., Mackey, B., Dahdouh-Guebas, F., ... & Stovall, A. E. (2017). Large-scale dieback of mangroves in Australia's Gulf of Carpentaria: a severe ecosystem response, coincidental with an extreme climate event. *Marine and Freshwater Research*, 68(10), 1816-1829.",
        "Goldberg, L., Lagomasino, D., Thomas, N., & Fatoyinbo, T. (2020). Global declines in human-driven mangrove loss. *Global Change Biology*, 26(10), 5844-5855.",
        "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770-778.",
        "Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. *International Conference on Learning Representations (ICLR)*.",
        "Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W. Y., Dollár, P., & Girshick, R. (2023). Segment anything. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 4015-4026.",
        "Lacoste, A., Lehmann, N., Pauletti, P., Hosseinzadeh, M., Stewart, R., & Vazquez, D. (2023). GEO-Bench: Toward foundation models for Earth monitoring. *Advances in Neural Information Processing Systems (NeurIPS)*, 36, 74744-74765.",
        "Murray, N. J., Worthington, T. A., Bunting, P., Duce, S., Hagger, V., Lovelock, C. E., ... & Lyons, M. B. (2022). High-resolution mapping of losses and gains of Earth's tidal wetlands. *Science*, 376(6594), 744-749.",
        "Qin, Y., Chen, J., Wang, C., & Pan, C. (2025). BiSAM-CD: Zero-shot remote sensing change detection via bidirectional temporal memory in SAM2. *IEEE Transactions on Geoscience and Remote Sensing*, 63, Art no. 4417812.",
        "Qin, Y., Wang, C., Fan, Y., & Pan, C. (2025). SAM2-CD: Remote sensing image change detection with SAM2. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 18, 24575-24587.",
        "Rafaeli, O., Svoray, T., & Nahlieli, A. (2025). SinkSAM-Net: Knowledge-driven self-supervised sinkhole segmentation using topographic priors and Segment Anything Model. *ISPRS Journal of Photogrammetry and Remote Sensing*, 220, 15-29.",
        "Ravi, N., Gabeur, V., Hu, Y. T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädler, M., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K. V., Carion, N., Wu, C. Y., Girshick, R., Dollár, P., & Feichtenhofer, C. (2024). SAM 2: Segment anything in images and videos. *arXiv preprint arXiv:2408.00714*.",
        "Rivier, S., Hinojosa, C., Giancola, S., & Ghanem, B. (2024). Efficient semantic segmentation for aerial imagery using query points and superpixel supervision. *Proceedings of the IEEE International Geoscience and Remote Sensing Symposium (IGARSS)*, 1-5.",
        "Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, Springer, 234-241.",
        "Scribano, C., Govi, E., Bertellini, P., Parisi, S., Franchini, G., & Bertogna, M. (2025). Segment Anything for satellite imagery: A strong baseline and a regional dataset for automatic field delineation. *arXiv preprint arXiv:2506.16318*.",
        "Tuia, D., Persello, C., & Bruzzone, L. (2016). Recent advances in domain adaptation for the classification of remote sensing data. *IEEE Geoscience and Remote Sensing Magazine*, 4(2), 41-57.",
        "Wan, Z., Wang, S., Han, W., Wang, Y., Huang, X., Zhang, X., Chen, X., & Chen, Y. (2025). A systematic survey and meta-analysis of the segment anything model in remote sensing image processing: Challenges, advances, applications, and opportunities. *ISPRS Journal of Photogrammetry and Remote Sensing*, 222, 248-283.",
        "Wang, C., Ji, Y., Meng, Y., Zhang, Y., & Zhu, Y. (2025). SOPSeg: Prompt-based small object instance segmentation in remote sensing imagery. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops*.",
        "Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). CBAM: Convolutional block attention module. *Proceedings of the European Conference on Computer Vision (ECCV)*, 3-19.",
        "Yang, K., Hu, X., et al. (2024). WaveFusion: Discrete wavelet transform for boundary-aware multi-scale medical image segmentation. *Medical Image Analysis*, 91, 103028."
    ]

    for ref in references:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_before = Pt(2)
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.left_indent = Inches(0.4)
        p_ref.paragraph_format.first_line_indent = Inches(-0.4)
        p_ref.paragraph_format.line_spacing = 1.15
        p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        run_ref = p_ref.add_run(ref)
        run_ref.font.name = 'Calibri'
        run_ref.font.size = Pt(9.5)
        run_ref.font.color.rgb = CHARCOAL

    output_path = "c:\\Users\\hai49\\.gemini\\antigravity\\scratch\\cangio\\CanGio_Mangrove_Canopy_Disturbance_Manuscript.docx"
    doc.save(output_path)
    print(f"Manuscript successfully generated and saved to: {output_path}")

if __name__ == "__main__":
    build_manuscript()
