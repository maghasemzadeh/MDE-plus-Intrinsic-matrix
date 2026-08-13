on run argv
    set inputPath to item 1 of argv
    set outputPath to item 2 of argv
    set outputHFS to (POSIX file outputPath) as text
    tell application "Microsoft Word"
        open POSIX file inputPath
        set docRef to active document
        save as docRef file name outputHFS file format format PDF
    end tell
end run
