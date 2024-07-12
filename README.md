Use these files _productively_ for educational purposes (to get some inspiration or to compare results) and, as always, at your own risk of course :)

1. Create a new Python project (e.g. via File > Project from Version Control... in PyCharm).
2. Place the project files, your `.hm` file and your `.xlsx` template in your project folder.
3. Replace `PROJECT_PATH` in `queryconfig.xml` with the path to your project folder (x7). Use `/`, not `\`.
4. Replace `STUDENT_ID` in `config.ini` with your student ID omitting the leading 0.
5. Run step 1 in `main.py`. Make sure your template isn't locked by Excel running in the background.
6. Enter the homogenized properties in HyperMesh (as well as all other problem data).
7. Run the analysis in HyperMesh. The solver deck (`.fem` file) should have the same name as the `.hm` file.
8. Import your results (`.h3d` file) in HyperMesh (File > Import > Results).
9. Query your results using the provided `queryconfig.xml` (Post > Query > Tools > Use XML).
10. Run step 2 in `main.py`.
