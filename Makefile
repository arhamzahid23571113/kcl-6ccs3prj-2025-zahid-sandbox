.PHONY: install smoke explain-smoke
install:
	pip install -e .

smoke:
	kcl-run-one-pixel untargeted --limit 6 --batch-size 64 --pop 96 --max-gens 15 --out tmp/smoke/de_untargeted.csv --save-adv-dir tmp/smoke/adv_untargeted
	kcl-run-one-pixel guided --limit 5 --batch-size 64 --pop 96 --max-gens 12 --rise-masks 400 --out tmp/smoke/de_guided.csv
	kcl-run-one-pixel targeted-sweep --limit 4 --batch-size 64 --targets-mode random --targets-per-image 3 --pop 96 --max-gens 12 --out tmp/smoke/de_targeted_sweep.csv --save-adv-dir tmp/smoke/adv_sweep
	kcl-run-one-pixel fastprior --limit 4 --batch-size 64 --pop 96 --max-gens 12 --xy-topk 16 --out tmp/smoke/de_fastprior.csv --save-adv-dir tmp/smoke/adv_fastprior
	kcl-run-one-pixel targeted-mt-prior --limit 4 --batch-size 64 --pop 96 --max-gens 12 --xy-topk 16 --out tmp/smoke/de_mt_prior.csv --save-adv-dir tmp/smoke/adv_mt_prior

explain-smoke:
	tail -n +2 tmp/smoke/de_untargeted.csv | cut -d',' -f2 | sed 's/[^0-9]//g' | sort -n | uniq | head -n 24 > tmp/smoke/indices_expl.txt
	kcl-explain --out tmp/smoke/explanations.csv --indices-file tmp/smoke/indices_expl.txt --adv-dir tmp/smoke/adv_untargeted --ig-steps 10 --rise-masks 120 --data-root ./data --device mps --saliency-device cpu
	kcl-sum-expl --expl tmp/smoke/explanations.csv --attacks tmp/smoke/de_untargeted.csv --adv-dir tmp/smoke/adv_untargeted --out tmp/smoke/summary_explanations.md
	kcl-sum-targeted --csv tmp/smoke/de_targeted_sweep.csv --out tmp/smoke/summary_targeted.md
