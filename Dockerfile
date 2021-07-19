FROM sarisia/texlive:2019
ADD thesis/ .
ENTRYPOINT [ "make", "build" ]
