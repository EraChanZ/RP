# Absolute paths from RP_Docs directory
use Cwd qw(abs_path);
use File::Basename;

# Get the root directory (where .latexmkrc is located)
$ROOT = dirname(abs_path($0));

# Define output and auxiliary directories
$out_dir = "$ROOT/pdf";
$aux_dir = "$ROOT/aux";

# Clean up policy
$clean_ext = 'aux bbl bcf blg brf idx ilg ind lof log lot out run.xml toc synctex.gz nav snm fdb_latexmk fls vrb';

# Force creation of output directories
mkdir($out_dir) unless -d $out_dir;
mkdir($aux_dir) unless -d $aux_dir;

# Ensure LaTeX writes outputs to the correct directories
$out_dir = $out_dir;
$aux_dir = $aux_dir;

# Optionally, you can specify additional settings or hooks here