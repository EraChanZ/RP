   #!/bin/bash

   # Define directories
   BUILD_DIR="RP_Docs/pdf"
   AUX_DIR="RP_Docs/aux"
   TEX_FILE="RP_Docs/tex/presentationNew.tex"

   # Ensure auxiliary directory exists
   mkdir -p "$AUX_DIR"

   # Run latexmk to compile the LaTeX project
   latexmk -pdf -interaction=nonstopmode -file-line-error -synctex=1 -cd -output-directory="$BUILD_DIR" "$TEX_FILE"

   # Check if latexmk succeeded
   if [ $? -eq 0 ]; then
       echo "LaTeX build successful."

       # Delete all non-.pdf files in the BUILD_DIR
       find "$BUILD_DIR" -type f ! -name "*.pdf" -exec rm -f {} +

       echo "Non-PDF auxiliary files have been deleted from $BUILD_DIR."
   else
       echo "LaTeX build failed. Check the logs for details."
       exit 1
   fi