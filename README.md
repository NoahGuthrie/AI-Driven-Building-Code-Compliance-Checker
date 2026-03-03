# AI-Driven-Building-Code-Compliance-Checker

Overview
This repository contains a vision-based parsing and optimization engine designed to automate the detection of building code violations in SVG floor plans. Rather than relying on rigid deterministic rules, this system uses probabilistic reasoning and Reinforcement Learning to resolve complex geometric conflicts and measurement uncertainties found in real-world architectural datasets.

Core Features
Probabilistic Vision Pipeline: Utilizes OpenCV to parse CubiCasa5k SVG data, incorporating Gaussian probabilistic reasoning to compute confidence scores for violations under measurement uncertainty.

Q-Learning Optimization Engine: Implements a constrained optimization framework to autonomously resolve geometric overlaps and code conflicts, achieving a 97.5% reduction in violations across benchmarked samples.

Production-Scale Performance: Features a batch processing framework capable of evaluating 1000+ plans with a sub-second latency (~0.2s per plan).

Automated Resolution: Employs a CSP (Constraint Satisfaction Problem) solver to validate and automatically resolve 82.5% of non-compliant cases based on international building standards.

Tech Stack
Core: Python, SciPy, NumPy

Vision: OpenCV

RL/Optimization: Q-Learning, Constraint Satisfaction Solvers

Data: CubiCasa5k Dataset

Technical Implementation Note
A key challenge addressed in this project was the high variance in raw SVG data. I engineered a custom ETL process to transform raw inference tensors into structured logs, allowing for retrospective performance analysis and model iteration. This focus on failure mode diagnosis ensured the system remained robust against the edge cases common in "dirty" real-world spatial data.
