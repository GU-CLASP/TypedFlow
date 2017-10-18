
viewdoc: dist/doc/html/typedflow/index.html
	xdg-open $<

dist/doc/html/typedflow/index.html:
	styx cabal -- haddock --hyperlink-source
	styx cabal -- hscolour

