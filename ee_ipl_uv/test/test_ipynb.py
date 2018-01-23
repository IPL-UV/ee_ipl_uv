import os
import subprocess
import tempfile
import nbformat
import inspect


def _notebook_run(path):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    # dirname, __ = os.path.split(path)
    # os.chdir(dirname)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter",
                "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=3600",
                "--output", fout.name, path]
        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = [output for cell in nb.cells if "outputs" in cell
              for output in cell["outputs"]
              if output.output_type == "error"]

    return nb, errors


def test_ipynb():
    path_script = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    files_ipy = ["band_covariance.ipynb",
                 "kernels.ipynb",
                 "kernels_part_2.ipynb",
                 "cloud_filter.ipynb",
                 "Normalization.ipynb",
                 "test_normalization.ipynb",
                 "convolutions.ipynb",
                 "test_kernel.ipynb",
                 "test_ridge_regression.ipynb"]

    for notebook in files_ipy:
        print("Testing :", notebook)
        nb, errors = _notebook_run(path_script+"/"+notebook)
        assert errors == []

