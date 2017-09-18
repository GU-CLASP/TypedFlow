test: s2s.py synthtrees.txt main.py
	nix-shell --run "python main.py"

s2s.py: Seq2Seq.hs
	nix-shell ../../.styx/shell.nix --run "ghci -i../.. Seq2Seq.hs -e main"

synthtrees.txt: GenTr.hs
	nix-shell ../../.styx/shell.nix --run "ghc --make GenTr"
	./GenTr
