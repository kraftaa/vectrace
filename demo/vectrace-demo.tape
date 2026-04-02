Output demo/vectrace-demo.gif

Set FontSize 16
Set Width 1080
Set Height 720
Set Theme "Catppuccin Mocha"
Set Padding 24

Type ".venv/bin/vectrace onboard --db /tmp/vectrace-demo.db --output /tmp/vectrace-demo.html"
Enter
Sleep 1s

Type ".venv/bin/vectrace trace --vector-id vec_demo_001 --collection support_kb --db /tmp/vectrace-demo.db"
Enter
Sleep 1.2s

Type ".venv/bin/vectrace trace --vector-id vec_demo_001 --collection support_kb --db /tmp/vectrace-demo.db --format json"
Enter
Sleep 1.2s

Type ".venv/bin/vectrace report --vector-id vec_demo_001 --collection support_kb --db /tmp/vectrace-demo.db --output /tmp/vectrace-demo.html --redact-preview"
Enter
Sleep 1s
