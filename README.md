# **Data Visualization of Suicide Mortality (2000-2019)**  

**Author:** Lily Gates  
**Date:** July 2024  

## **Description**  
This project visualizes trends in global suicide mortality rates from 2000 to 2019. The analysis compares geographic and economic regions across various demographic categories. Key visual outputs include:  

1. **Stacked Ladder Plot:**  
   - Title: "Comparing Annual Suicide Mortality Rates (2000-2019)"  
   - Compares mortality rates across different geographical regions and globally.  

2. **Stacked Line Plot:**  
   - Title: "Comparing Annual Suicide Mortality Rates (2000-2019) By Sexes"  
   - Displays rates for different sexes using a 5-year running mean.  

3. **Histograms:**  
   - A histogram for each year from 2000 to 2019 is generated and saved in the `hist_plots` directory.  
   - These plots can be combined into an animated GIF using ImageMagick’s `convert` command.  

---

## **Project Structure**  

- **`final_hist.py`:**  
  Generates and saves yearly histograms labeled by year in the `hist_plots` directory.  

  **Key Features:**  
  - Saves plots as PNG files using:  
    ```python
    fig.savefig(f"../hist_plots/plot_{year}")
    ```  
  - Suitable for creating an animated GIF from histogram PNGs.  

- **`final_ladder.py`:**  
  Generates ladder plots comparing annual suicide mortality rates between 2000 and 2019.  

---

## **Dependencies**  

Ensure the following Python modules are installed:  

- `cartopy`  
- `glob`  
- `h5py`  
- `matplotlib`  
- `numpy`  
- `os`  
- `pandas`  
- `datetime`  

Install dependencies using:  

```bash
pip install cartopy h5py matplotlib numpy pandas
```

---

## **How to Run the Program**  

1. Clone the repository or download the project files.  
2. Ensure the necessary dependencies are installed.  
3. Run `final_hist.py` and `final_ladder.py` to generate plots and histograms:  

   ```bash
   python final_hist.py  
   python final_ladder.py
   ```
   
Use ImageMagick to convert histogram PNGs to an animated GIF (optional):

```bash
convert ../hist_plots/plot_*.png suicide_histogram_animation.gif
```
   
## Future Features
* Future trend predictions based on demographic data, using statistical or machine learning methods to forecast trends in suicide mortality rates.
* Heatmap visualizations using Cartopy maps to represent regional data trends over time, providing a spatial perspective on how suicide mortality rates differ across geographic regions.
   