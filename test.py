import os
import random

import polars as pl
import torch
import torch.nn.functional as F

from model import test_model


INPUT_DIM = 27
HIDDEN_DIM = 64
EOS_IDX = 26
BEST_PATH = "best.pt"
TEST_SAMPLES = 200
INPUT_DIM = 27
HIDDEN_DIM = 440
EOS_IDX = 26
BATCH_SIZE = 512*8
LEARNING_RATE = 0.001
EPOCHS = 500

VAL_SPLIT = 0.1
USE_BEST = True
BEST_PATH = "best.pt"
LR_FACTOR = 0.7
LR_PATIENCE = 10
LR_MIN = 1e-6

TEST_WORDS = [
    "asynchronous", "polymorphism", "encapsulation", "architecture", "concurrency",
    "parallelism", "inheritance", "recursion", "scalability", "distributed",
    "microservices", "orchestration", "deployment", "infrastructure", "configuration",
    "optimization", "performance", "throughput", "latency", "bandwidth",
    "bottleneck", "deadlock", "synchronization", "transaction", "consistency",
    "availability", "partition", "redundancy", "resiliency", "observability",
    "instrumentation", "monitoring", "telemetry", "visualisation", "abstraction",
    "dependency", "injection", "singleton", "prototype", "decorator",
    "mitochondria", "photosynthesis", "metabolism", "homeostasis", "evolutionary",
    "biodiversity", "ecosystem", "environment", "atmospheric", "stratosphere",
    "thermosphere", "ionosphere", "magnetosphere", "lithosphere", "hydrosphere",
    "geochemical", "biophysical", "neurological", "cardiovascular", "respiratory",
    "immunological", "endocrinology", "pharmacology", "toxicology", "pathological",
    "therapeutic", "diagnostic", "radiological", "biotechnology", "nanotechnology",
    "cybernetics", "cryptography", "blockchain", "consensus", "encryption",
    "decryption", "authentication", "authorisation", "vulnerability", "remediation",
    "existentialism", "phenomenology", "epistemology", "metaphysics", "ontology",
    "utilitarianism", "structuralism", "nihilism", "pragmatism", "rationalism",
    "empiricism", "stoicism", "humanism", "individualism", "collectivism",
    "quintessential", "supercilious", "pusillanimous", "magnanimous", "fastidious",
    "capricious", "surreptitious", "ephemeral", "clandestine", "pernicious",
    "ubiquitous", "omnipresent", "ambivalent", "equivocal", "mercurial",
    "phlegmatic", "sanguine", "melancholic", "choleric", "recalcitrant",
    "intransigent", "incendiary", "provocative", "inflammatory", "contentious",
    "argumentative", "adversarial", "antagonistic", "belligerent", "pugnacious",
    "synchronous", "asymmetrical", "symmetrical", "proportional", "logarithmic",
    "exponential", "stochastic", "probabilistic", "deterministic", "heuristic",
    "algorithmic", "mathematical", "geometrical", "topological", "arithmetic",
    "calculus", "differential", "integral", "derivative", "trigonometry",
    "bureaucracy", "aristocracy", "democracy", "autocracy", "technocracy",
    "meritocracy", "plutocracy", "oligarchy", "anarchy", "sovereignty",
    "jurisdiction", "legislative", "executive", "judicial", "constitutional",
    "parliamentary", "diplomatic", "international", "globalization", "privatization",
    "nationalization", "liberalization", "regulation", "deregulation", "intervention",
    "preliminary", "subsidiary", "supplementary", "additional", "fundamental",
    "elementary", "intermediate", "advanced", "sophisticated", "complicated",
    "convoluted", "intricate", "detailed", "comprehensive", "exhaustive",
    "preparatory", "introductory", "concluding", "retrospective", "prospective",
    "hypothetical", "theoretical", "empirical", "analytical", "synthetic",
    "quantitative", "qualitative", "conceptual", "categorical", "systematic",
    "methodological", "chronological", "geographical", "statistical", "demographic",
    "fluctuation", "variation", "oscillation", "vibration", "resonance",
    "equilibrium", "stability", "instability", "turbulence", "vorticity",
    "viscosity", "buoyancy", "velocity", "acceleration", "momentum",
    "inertia", "gravity", "friction", "resistance", "impedance",
    "capacitance", "inductance", "conductivity", "permittivity", "permeability",
    "transparency", "opacity", "reflectivity", "refraction", "diffraction",
    "interference", "polarization", "frequency", "amplitude", "wavelength",
    "spectrum", "radiation", "convection", "conduction", "emissivity",
    "collaboration", "integration", "coordination", "synchronisation", "standardization",
    "harmonization", "modernization", "digitalization", "transformation", "innovation",
    "development", "implementation", "maintenance", "operation", "management",
    "leadership", "strategy", "tactical", "operational", "administrative",
    "personnel", "recruitment", "procurement", "logistics", "distribution",
    "marketing", "advertising", "promotional", "sponsorship", "commercial",
    "industrial", "agricultural", "financial", "economical", "monetary",
    "statistical", "mathematical", "scientific", "biological", "chemical",
    "physical", "geological", "astronomical", "meteorological", "ecological",
    "environmental", "sociological", "psychological", "anthropological", "historical",
    "linguistic", "philosophical", "theological", "educational", "technological",
    "mechanical", "electrical", "electronic", "structural", "functional",
    "dynamic", "static", "kinetic", "potential", "thermal",
    "nuclear", "atomic", "molecular", "cellular", "genetic",
    "hereditary", "chromosomal", "ribosomal", "cytoplasmic", "mitotic",
    "meiotic", "metabolic", "enzymatic", "hormonal", "physiological",
    "anatomical", "histological", "embryological", "evolutionary", "phylogenetic",
    "taxonomy", "species", "genus", "family", "order",
    "class", "phylum", "kingdom", "domain", "organism",
    "population", "community", "ecosystem", "biosphere", "habitat",
    "niche", "adaptation", "selection", "mutation", "variation",
]


def load_words(path="words.parquet"):
	df = pl.read_parquet(path)
	return df["words"].to_list()


def build_vocab():
	alphabet = "abcdefghijklmnopqrstuvwxyz "
	return {char: idx for idx, char in enumerate(alphabet)}


def decode_indices(indices, idx_to_char):
	chars = []
	for idx in indices:
		if idx == EOS_IDX:
			break
		chars.append(idx_to_char[idx])
	return "".join(chars)


def encode_word(word, char_to_idx):
	seq = f"{word} "
	if any(char not in char_to_idx for char in seq):
		return None
	indices = [char_to_idx[char] for char in seq]
	return torch.tensor(indices, dtype=torch.long)


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if not os.path.exists(BEST_PATH):
		print(f"{BEST_PATH} not found. Train first before testing.")
		return

	if TEST_WORDS:
		words = list(TEST_WORDS)
		max_samples = len(words)
	else:
		words = load_words()
		random.shuffle(words)
		max_samples = TEST_SAMPLES
	char_to_idx = build_vocab()
	idx_to_char = [char for char, _ in sorted(char_to_idx.items(), key=lambda item: item[1])]

	model = test_model(input_size=INPUT_DIM, hidden_size=HIDDEN_DIM, eos_index=EOS_IDX)
	model.to(device)
	model.eval()

	try:
		state = torch.load(BEST_PATH, map_location=device)
		model.load_state_dict(state)
	except (OSError, RuntimeError, ValueError) as exc:
		print(f"Could not load {BEST_PATH}: {exc}")
		return

	total_tokens = 0
	correct_tokens = 0
	class_tp = [0] * INPUT_DIM
	class_actual = [0] * INPUT_DIM

	tested = 0
	with torch.no_grad():
		for word in words:
			encoded = encode_word(word, char_to_idx)
			if encoded is None:
				continue

			seq_len = encoded.numel()
			one_hot = F.one_hot(encoded, num_classes=INPUT_DIM).to(dtype=torch.float32)
			one_hot = one_hot.unsqueeze(0).to(device)

			logits, _, __ = model(one_hot, max_length=seq_len, stop_on_eos=False)
			preds = logits.argmax(dim=-1).squeeze(0).cpu()
			pred_word = decode_indices(preds.tolist(), idx_to_char)

			print(f"IN: {word} | OUT: {pred_word}")

			for target_idx, pred_idx in zip(encoded.tolist(), preds.tolist()):
				total_tokens += 1
				class_actual[target_idx] += 1
				if pred_idx == target_idx:
					correct_tokens += 1
					class_tp[target_idx] += 1

			tested += 1
			if tested >= max_samples:
				break

	if total_tokens == 0:
		print("No valid samples found for testing.")
		return

	accuracy = correct_tokens / total_tokens

	per_class_recall = []
	for cls in range(INPUT_DIM):
		if class_actual[cls] == 0:
			continue
		per_class_recall.append(class_tp[cls] / class_actual[cls])

	macro_recall = sum(per_class_recall) / len(per_class_recall) if per_class_recall else 0.0
	micro_recall = accuracy

	print(f"Tested words: {tested}")
	print(f"Token Accuracy: {accuracy:.4f}")
	print(f"Macro Recall: {macro_recall:.4f}")
	print(f"Micro Recall: {micro_recall:.4f}")


if __name__ == "__main__":
	main()
