#!/usr/bin/perl

$latex = 'uplatex -synctex=1 %O %S';
$bibtex = 'pbibtex %O %B';
$dvipdf = 'dvipdfmx -f uptex-ipaex.map %O %S';
$pdf_mode = 3;
