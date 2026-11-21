# Opens the thesis in Microsoft Word, converts linear-format formulas into Word equations,
# formats and updates the table of contents and all fields, saves the document, exports a
# PDF next to it and prints the page count.
param(
    [string]$Docx = (Join-Path $PSScriptRoot "Магістерська_робота.docx")
)
$ErrorActionPreference = "Stop"
$Docx = (Resolve-Path $Docx).Path
$Pdf = [System.IO.Path]::ChangeExtension($Docx, ".pdf")
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx)

    # Equations: paragraphs "<TAB>formula<TAB>(N.N)" built by build_docx.py.
    $eqs = 0
    foreach ($p in @($doc.Paragraphs)) {
        $text = $p.Range.Text
        if ($text -match "^\t(.+)\t\(\d+\.\d+\)\r$") {
            $start = $p.Range.Start + 1
            $range = $doc.Range($start, $start + $Matches[1].Length)
            $math = $doc.OMaths.Add($range).OMaths.Item(1)
            $math.BuildUp()
            $math.Range.Font.Name = "Cambria Math"
            $math.Range.Font.Size = 14
            $eqs++
        }
    }

    # TOC 1..3 styles (wdStyleTOC1 = -20): left aligned, no first-line indent.
    $indents = @(0, 17, 34)
    for ($i = 0; $i -lt 3; $i++) {
        $style = $doc.Styles.Item(-20 - $i)
        $style.Font.Name = "Times New Roman"
        $style.Font.Size = 14
        $style.ParagraphFormat.Alignment = 0
        $style.ParagraphFormat.FirstLineIndent = 0
        $style.ParagraphFormat.LeftIndent = $indents[$i]
        $style.ParagraphFormat.SpaceAfter = 0
        $style.ParagraphFormat.LineSpacingRule = 1   # 1.5 lines
    }

    $doc.Repaginate()
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Fields.Update() | Out-Null
    $doc.Repaginate()
    foreach ($toc in $doc.TablesOfContents) { $toc.UpdatePageNumbers() }
    $pages = $doc.ComputeStatistics(2)
    $doc.Save()
    $doc.ExportAsFixedFormat($Pdf, 17)
    $doc.Close()
    Write-Output "EQUATIONS=$eqs"
    Write-Output "PAGES=$pages"
} finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
}
