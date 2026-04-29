.PHONY: help install test score figures hf-pack split clean anon_zip

help:
	@echo "DECEPT-Bench targets:"
	@echo "  install        Install runtime deps (pip install -r requirements.txt)"
	@echo "  test           Run scorer test suite"
	@echo "  score MODEL=X  Score data/$$MODEL/ and write data/$$MODEL/scores.json"
	@echo "  figures        Regenerate paper/figures/* from current scores"
	@echo "  hf-pack        Package items as HF dataset (data/hf_dataset/)"
	@echo "  split DIR=X    Apply 70/30 public-dev / private-test split to data/$$DIR"
	@echo "  clean          Remove pyc, __pycache__, /tmp/*"
	@echo "  anon_zip       Build submission.zip with author/repo identity scrubbed"
	@echo "                 (for double-blind NeurIPS supplement upload)"

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v -p no:anchorpy

score:
	@test -n "$(MODEL)" || (echo "set MODEL=<dirname under data/>"; exit 1)
	python src/decept_score_v1.py --data_dir data/$(MODEL) --out_dir data/$(MODEL)

figures:
	python src/decept_figures.py

hf-pack:
	python src/decept_hf_dataset.py --out_dir data/hf_dataset

split:
	@test -n "$(DIR)" || (echo "set DIR=<dirname under data/>"; exit 1)
	python src/decept_split.py --in_dir data/$(DIR) --out_dir data/decept-v1-$(DIR)

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# Build a double-blind-anonymized submission.zip for the NeurIPS supplement
# upload. Stages a copy of the repo to /tmp/, scrubs author + repo identity
# from README / CITATION / supplements, drops .git and other obvious
# identity-leaking files, then zips the staged tree. Live repo is NEVER
# modified — open submission.zip to verify before upload.
ANON_STAGE := /tmp/decept-bench-anon
anon_zip:
	@command -v rsync >/dev/null || (echo "rsync required"; exit 1)
	@command -v zip   >/dev/null || (echo "zip required"; exit 1)
	rm -rf $(ANON_STAGE) submission.zip
	mkdir -p $(ANON_STAGE)
	# Stage everything except identity-leaking + bulky build artifacts.
	# .git/ is the biggest leak (history reveals author + repo URL).
	rsync -a \
	  --exclude='.git/' --exclude='.github/' \
	  --exclude='internal/' \
	  --exclude='__pycache__/' --exclude='*.pyc' \
	  --exclude='.DS_Store' --exclude='.pytest_cache/' \
	  --exclude='.venv/' --exclude='venv/' --exclude='env/' \
	  --exclude='paper/main.aux' --exclude='paper/main.log' \
	  --exclude='paper/main.bbl' --exclude='paper/main.blg' \
	  --exclude='paper/main.out' --exclude='paper/main.toc' \
	  --exclude='paper/template.zip' \
	  --exclude='README_PUBLIC.md' \
	  --exclude='data/em_datasets/' --exclude='data/v0_legacy/' \
	  --exclude='data/smoke/' --exclude='data/decept-v1-*/' \
	  --exclude='data/seed_var/' --exclude='data/decept_seed_var/' \
	  --exclude='data/hf_dataset/' \
	  --exclude='*.bak' \
	  --exclude='*/test/' \
	  --exclude='private_test/' \
	  --exclude='.anon_patterns' \
	  --exclude='.anon_patterns.example' \
	  --exclude='paper/main.synctex.gz' \
	  --exclude='paper/sections/.tex.swp' \
	  --exclude='runs/' --exclude='logs/' --exclude='*.log' \
	  --exclude='submission.zip' \
	  ./ $(ANON_STAGE)/
	# Scrub identity from human-readable files.
	# Replace github URL → URL-withheld; replace author → Anonymous.
	# Files we know contain identity: README.md, HIGHLIGHTS.md (citation block),
	# CITATION.cff, data/hf_dataset/README.md, SPEC_V1.md, SUBMISSIONS.md.
	# We sed across all *.md, *.cff, *.tex, *.yml/yaml just to be safe.
	# Apply identity substitutions from .anon_patterns (gitignored). If
	# the file is missing we fall back to a generic github-URL scrub so
	# the target still produces something usable; the user must then
	# manually grep the staged tree for any remaining identity tokens.
	if [ -f .anon_patterns ]; then \
	  echo "  applying .anon_patterns sed rules"; \
	  find $(ANON_STAGE) \( -name '*.md' -o -name '*.cff' -o -name '*.tex' \
	                      -o -name '*.yml' -o -name '*.yaml' -o -name '*.toml' \
	                      -o -name '*.txt' -o -name '*.json' \
	                      -o -name '*.py'  -o -name 'Makefile' \) \
	    -exec sed -i.bak -f .anon_patterns {} \; ; \
	else \
	  echo "  .anon_patterns missing — applying generic github-URL fallback only"; \
	  echo "  copy .anon_patterns.example -> .anon_patterns and customise for full scrub"; \
	  find $(ANON_STAGE) \( -name '*.md' -o -name '*.cff' -o -name '*.tex' \
	                      -o -name '*.yml' -o -name '*.yaml' -o -name '*.toml' \
	                      -o -name '*.txt' -o -name '*.json' \
	                      -o -name '*.py'  -o -name 'Makefile' \) \
	    -exec sed -i.bak \
	      -e 's|https://github.com/[A-Za-z0-9_-]*/decept-bench|<repo-URL-withheld-for-double-blind-review>|g' \
	      -e 's|github.com/[A-Za-z0-9_-]*/decept-bench|<repo-URL-withheld-for-double-blind-review>|g' \
	      {} \; ; \
	fi
	find $(ANON_STAGE) -name '*.bak' -delete
	@echo ""; echo "==== anonymity sanity check ===="
	@echo "  manually grep the staged tree before zipping. Common patterns:"
	@echo "    grep -RIE 'github\\.com/[A-Za-z0-9_-]+/decept-bench' $(ANON_STAGE)"
	@echo "    grep -RIE '<your-name>|<your-email>|<your-username>' $(ANON_STAGE)"
	@echo ""
	# Build the submission zip.
	cd $(ANON_STAGE) && zip -qrX $(CURDIR)/submission.zip . -x ".*"
	@du -h submission.zip
	@echo ""
	@echo "Submission archive ready: submission.zip"
	@echo "Staged tree: $(ANON_STAGE)/"
	@echo "Recommended next step: spot-check by extracting to a fresh dir and"
	@echo "  grepping for any identity tokens you know are in your environment."
