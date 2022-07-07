# Criteo Privacy-preserving Machine Learning Competition @ AdKDD'21

In the Criteo-AdKdd'21 competition, the goal was to learn a ML model from aggregated data.
- You can access the competition at http://go.criteo.net/criteo-privacy-preserving-ml-competition
- You can find an article summirizing the results and providing additional hinsights at https://arxiv.org/pdf/2201.13123.pdf
- 
This repository contains useful code and documentation for the challenge, and the notebooks to run the experiments reported in the research article.

## Related Works

`RELATEDWORKS.md` contains a short selection of links to potentially interesting works relating Privacy and ML.
Please feel free to submit pull requests to enhance this selection.

## Fetch the data
In the `data` directory you can find two notebooks:
- `fetch_datasets`: allows to retieve all the competition data as well as the granular data used to make the aggregations and another larger test set, used in our experiments to evaluate the performance of the models according to the number of samples.
- `generate_aggregated_datasets`: notebook that we used to generate the aggregations. It is necessary to run it to create the noiseless aggregated datasets that we use in the experiments to study the robustness to noise of the different methods.

## Running experiments

### Gradient Boosting Trees

Models and experiments performed with Gradient Boosting Trees can be found in the `gbt` folder. There are two notebooks, one for clicks and one for sales.

### Logistic Regression

Models and experiments performed with a Logistic Regression can be found in the `logistic_regression` folder in the `main` notebook. `encode_data` notebook should be run first to one-hot encode in a common space the modalities of each feature and the pairs of modalities of each pair of feature.
