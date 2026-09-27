🔹 Basic Questions
What is EvacuAI?
What problem does EvacuAI solve?
What are the main technologies used in EvacuAI?
What are the four crowd risk levels?
What is the main motivation behind EvacuAI?
🔹 Technical RAG Test
How does YOLOv8 contribute to EvacuAI?
How is crowd density calculated?
What is Lucas–Kanade optical flow used for?
What are Motion, Flow Conflict, and Turbulence?
How is the composite risk score calculated?
What are the weights used for a bottleneck ROI?
What are the thresholds for Low, Medium, High, and Critical risk?

Paper specifically defines the five-signal risk score and its ROI-dependent weights on page 6.

🔹 Numerical Questions — VERY IMPORTANT

Ye tumhare RAG ko actually test karne ke liye best hain:

What mAP@0.5 did EvacuAI achieve on the Indian Festival dataset?
What was the average precision and recall across the evaluation datasets?
What FPS did the system achieve using an NVIDIA RTX 3060?
What was the alert latency?
By how much did multi-source camera fusion reduce false negatives?
What was the false-positive SMS reduction?

Paper ke results mein Indian Festival dataset ka mAP@0.5 0.94, average mAP@0.5 0.917, RTX 3060 throughput 27 FPS, aur alert latency 80–100 ms report ki gayi hai.

🔹 Multi-step / Reasoning Questions
Explain the complete EvacuAI pipeline from camera input to alert generation.
Why does EvacuAI use both crowd density and optical flow instead of relying only on person detection?
How does the system distinguish an orderly high-density crowd from a potentially dangerous crowd?
Explain how a Critical risk condition leads to a Twilio SMS alert.
Why is the 15-second Critical persistence filter used?
Explain how multi-camera fusion improves the system.
What are the limitations of EvacuAI mentioned in the paper?
