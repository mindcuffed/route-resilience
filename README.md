\# Route Resilience



\## Occlusion-Robust Road Extraction and Graph-Theoretic Criticality Analysis for Urban Mobility



Route Resilience is a research prototype that extracts roads from satellite imagery, repairs fragmented road masks, converts them into a graph, identifies critical road segments, and simulates network failures.



The project focuses on situations where roads are partially hidden by trees, shadows, buildings, or other occlusions.



\---



\## Problem Statement



Satellite-based road extraction often produces broken and disconnected road masks. These masks are not sufficient for urban mobility analysis because roads must be represented as a connected network.



A disconnected road mask can lead to:



\- Incorrect route connectivity

\- Missing intersections

\- Incorrect identification of critical roads

\- Unreliable failure simulations



\---



\## Objective



The objective of this project is to develop a prototype pipeline that:



1\. Extracts roads from satellite imagery.

2\. Produces a binary road mask.

3\. Repairs gaps and fragmented road segments.

4\. Converts the mask into a graph.

5\. Identifies critical nodes and road segments.

6\. Simulates random and targeted road failures.

7\. Calculates network resilience.



\---



\## Complete Pipeline



```text

Satellite Image

&#x20;     ↓

Image Normalization

&#x20;     ↓

U-Net Road Segmentation

&#x20;     ↓

Probability Thresholding

&#x20;     ↓

Gap and Occlusion Repair

&#x20;     ↓

Road Graph Construction

&#x20;     ↓

Criticality Analysis

&#x20;     ↓

Failure Simulation

&#x20;     ↓

Resilience Score

