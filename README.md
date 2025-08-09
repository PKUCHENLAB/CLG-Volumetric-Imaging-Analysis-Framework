Welcome to Zebrafish Project!







`git clone https://github.com/HKUJUNE/ZebrafishProject.git`

`cd main`

`micromamba env create -f environment.yml -n zebragdm`

`micromamba activate zebragdm`



If you want to compute the correlation matrix of your time-series data and choose a threshold based on the distribution of correlations, please run the code file `ZebrafishProject\main\corrdistributionon.py`. In this script, we provide the distribution lines for four percentiles (95 %, 90 %, 85 %, 80 %) as candidate thresholds, and you will also obtain the corresponding visualization. You can choose an appropriate threshold and correlation-value rule according to the observed distribution to construct your network.

`python corrdistributionon.py`

Once you have selected an appropriate threshold and correlation-value rule for building your network, proceed by running the code file `ZebrafishProject\main\bulidyournetwork.py`. In the script, set the input and output paths for your data file, specify your chosen rule for selecting correlation values, and define the threshold you have decided on. Then execute the script. We have also included network-visualization output file in the code. You can use the generated network plot to verify whether your chosen threshold and correlation-value rule meet your expectations.

`python bulidyournetwork.py`

If you are satisfied with the network you have built, it’s time to start dismantling it! We provide several code files to help you dismantle your network ( `ZebrafishProject\main\dismantling_XXX.py `), each runnable on either CPU or GPU. Specifically, we offer dismantling strategies based on degree centrality, betweenness centrality, and a new method—`zebragdm`—that incorporates multiple optimizations on Marco Grassia et al.’s GDM framework (Machine-learning dismantling and early-warning signals of disintegration in complex systems. *Nature Communications*, 2021, 12(1): 5190). If you wish to use a dismantling method that incorporates multiple metrics, please apply the `zebragdm` model to your data, and you will need to adjust the relevant parameters in the code. Conversely, if you opt for a single-metric dismantling method, no parameter adjustments are necessary. Choose the approach that best suits your research question, and set your dismantling target value directly in the code file. The final outputs will include detailed information on the dismantled nodes and a visualization of the dismantling process.

`python dismantling_XXX.py`

`Tip: GPU acceleration can speed up dismantling, but it incurs additional cost. If your network is small—e.g., only a few hundred nodes—using the CPU implementation is usually the better choice.`







