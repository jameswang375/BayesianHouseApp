import marimo

__generated_with = "0.16.2"
app = marimo.App(
    width="medium",
    app_title="Bayesian App for Real Estate",
    css_file="style.css",
    html_head_file="head.html",
)


@app.cell
def _():
    import marimo as mo
    import duckdb
    import pandas as pd
    import os
    import numpy as np
    from sqlmodel import Field, Session, SQLModel, create_engine, select
    from typing import Optional
    return (
        Field,
        Optional,
        SQLModel,
        Session,
        create_engine,
        duckdb,
        mo,
        np,
        os,
        pd,
        select,
    )


@app.cell
def _(duckdb):
    con = duckdb.connect("sqlmesh/houses.db")
    return (con,)


@app.cell
def _(con):
    df_sqlmodel = con.execute("SELECT * FROM sqlmesh.select_features").fetchdf()
    return (df_sqlmodel,)


@app.cell
def _(con):
    con.close()
    return


@app.cell
def _(Field, Optional, SQLModel):
    class Housing(SQLModel, table=True):
        __table_args__ = {"extend_existing": True}
        pid: Optional[int] = Field(default=None, alias="PID", primary_key=True)
        overall_qual: Optional[int] = Field(alias="Overall Qual", default=0)
        gr_liv_area: Optional[int] = Field(alias="Gr Liv Area", default=0)
        first_flr_sf: Optional[int] = Field(alias="1st Flr SF", default=0)
        year_built: Optional[int] = Field(alias="Year Built", default=0)
        year_remod_add: Optional[int] = Field(alias="Year Remod/Add", default=0)
        lot_area: Optional[int] = Field(alias="Lot Area", default=0)
        overall_cond: Optional[int] = Field(alias="Overall Cond", default=0)
        garage_area: Optional[int] = Field(alias="Garage Area", default=0)
        total_bsmt_sf: Optional[int] = Field(alias="Total Bsmt SF", default=0)
        full_bath: Optional[int] = Field(alias="Full Bath", default=0)
        sale_price: Optional[int] = Field(alias="SalePrice")
    return (Housing,)


@app.cell
def _(Housing, SQLModel, Session, create_engine, df_sqlmodel, os):
    os.makedirs("sqlmodel", exist_ok=True)
    if not os.path.exists("sqlmodel/houses_crud.db"):
        sqlite_file_name = "sqlmodel/houses_crud.db"
        sqlite_url = f"sqlite:///{sqlite_file_name}"
        engine = create_engine(sqlite_url, echo=True)

        SQLModel.metadata.create_all(engine)

        houses = []

        for _, row in df_sqlmodel.iterrows():
            record = Housing(
                pid=row["pid"],
                overall_qual=row["overall_qual"],
                gr_liv_area=row["gr_liv_area"],
                first_flr_sf=row["first_flr_sf"],
                year_built=row["year_built"],
                year_remod_add=row["year_remod_add"],
                lot_area=row["lot_area"],
                overall_cond=row["overall_cond"],
                garage_area=row["garage_area"],
                total_bsmt_sf=row["total_bsmt_sf"],
                full_bath=row["full_bath"],
                sale_price=row["sale_price"],
            )

            houses.append(record)

        with Session(engine) as session:
            session.add_all(houses)
            session.commit()

    else:
        sqlite_file_name = "sqlmodel/houses_crud.db"
        sqlite_url = f"sqlite:///{sqlite_file_name}"
        engine = create_engine(sqlite_url, echo=True)
    return (engine,)


@app.cell
def _():
    target_std = 0.4073182750630981
    target_mean = 12.020431074250771
    return target_mean, target_std


@app.cell
def _(np):
    _FEATURE_COLS = ['overall_qual', 'gr_liv_area', 'first_flr_sf', 'year_built',
                     'year_remod_add', 'lot_area', 'overall_cond', 'garage_area',
                     'total_bsmt_sf', 'full_bath']
    _SKIP_ROWS = [956, 1497, 1569, 2070, 2114, 2179, 2180]

    _posteriors = np.load("posterior_samples.npy")  # (800, 2921)

    import duckdb as _duckdb
    _con = _duckdb.connect("sqlmesh/houses.db")
    _df = _con.execute("SELECT * FROM sqlmesh.select_features").fetchdf()
    _con.close()

    _df_train = _df.dropna(subset=['garage_area', 'total_bsmt_sf']).reset_index(drop=True)
    _df_train = _df_train.drop(index=_SKIP_ROWS).reset_index(drop=True)

    _X = _df_train[_FEATURE_COLS].values.astype(float)
    feat_means = _X.mean(axis=0)
    feat_stds = _X.std(axis=0)
    _X_std = (_X - feat_means) / feat_stds

    posterior_weights, _, _, _ = np.linalg.lstsq(_X_std, _posteriors.T, rcond=None)
    _residuals = _posteriors.T - _X_std @ posterior_weights
    noise_std = float(_residuals.std())
    return feat_means, feat_stds, posterior_weights, noise_std


@app.cell
def _(feat_means, feat_stds, mo, noise_std, np, posterior_weights, target_mean, target_std):
    def predict_posterior(feature_values):
        x = np.array(feature_values, dtype=float)
        x_std = (x - feat_means) / feat_stds
        samples = x_std @ posterior_weights + np.random.normal(0, noise_std, 800)
        return np.exp(samples * target_std + target_mean)

    def decision(feature_values, asking_price):
        posterior = predict_posterior(feature_values)
        predicted_mean = float(posterior.mean())
        expected_margin = round(predicted_mean - asking_price, 2)
        p_worth_more = float((posterior > asking_price).mean())
        pct = round(p_worth_more * 100, 1)

        if p_worth_more > 0.55:
            return mo.callout(mo.md(f"<h3 style='text-align: left;'>✅ Buy. The model estimates a {pct}% probability this house is worth more than the asking price. Expected gain: ${expected_margin:,.0f}.</h3>"), kind="success")
        elif p_worth_more >= 0.45:
            return mo.callout(mo.md(f"<h3 style='text-align: left;'>🤔 Fair Purchase. The model estimates a {pct}% probability this house is worth more than the asking price. Expected margin: ${expected_margin:,.0f}.</h3>"), kind="warn")
        else:
            return mo.callout(mo.md(f"<h3 style='text-align: left;'>🛑 Don't Buy. The model estimates only a {pct}% probability this house is worth more than the asking price. Expected loss: ${abs(expected_margin):,.0f}.</h3>"), kind="danger")
    return decision, predict_posterior


@app.cell
def _(feat_means, mo):
    _m = feat_means.astype(int)
    input_overall_qual = mo.ui.slider(start=1, stop=10, step=1, value=int(round(feat_means[0])), label="Overall Quality (1–10)")
    input_gr_liv_area = mo.ui.number(start=0, stop=10000, value=int(feat_means[1]), label="Ground Living Area (sq ft)")
    input_first_flr_sf = mo.ui.number(start=0, stop=10000, value=int(feat_means[2]), label="1st Floor Area (sq ft)")
    input_year_built = mo.ui.slider(start=1870, stop=2010, step=1, value=int(round(feat_means[3])), label="Year Built")
    input_year_remod = mo.ui.slider(start=1950, stop=2010, step=1, value=int(round(feat_means[4])), label="Year Remodeled")
    input_lot_area = mo.ui.number(start=0, stop=500000, value=int(feat_means[5]), label="Lot Area (sq ft)")
    input_overall_cond = mo.ui.slider(start=1, stop=10, step=1, value=int(round(feat_means[6])), label="Overall Condition (1–10)")
    input_garage_area = mo.ui.number(start=0, stop=2000, value=int(feat_means[7]), label="Garage Area (sq ft)")
    input_total_bsmt_sf = mo.ui.number(start=0, stop=6000, value=int(feat_means[8]), label="Total Basement Area (sq ft)")
    input_full_bath = mo.ui.slider(start=0, stop=5, step=1, value=int(round(feat_means[9])), label="Full Bathrooms")
    asking_price = mo.ui.number(start=0, stop=10000000, value=150000, label="Asking Price ($)")
    return (
        asking_price,
        input_first_flr_sf,
        input_full_bath,
        input_garage_area,
        input_gr_liv_area,
        input_lot_area,
        input_overall_cond,
        input_overall_qual,
        input_total_bsmt_sf,
        input_year_built,
        input_year_remod,
    )


@app.cell
def _(
    asking_price,
    decision,
    input_first_flr_sf,
    input_full_bath,
    input_garage_area,
    input_gr_liv_area,
    input_lot_area,
    input_overall_cond,
    input_overall_qual,
    input_total_bsmt_sf,
    input_year_built,
    input_year_remod,
    mo,
    set_decision_state,
):
    def calling_decision(*_):
        feature_values = [
            input_overall_qual.value,
            input_gr_liv_area.value,
            input_first_flr_sf.value,
            input_year_built.value,
            input_year_remod.value,
            input_lot_area.value,
            input_overall_cond.value,
            input_garage_area.value,
            input_total_bsmt_sf.value,
            input_full_bath.value,
        ]
        if asking_price.value and asking_price.value > 0:
            set_decision_state(decision(feature_values, asking_price.value))
        else:
            set_decision_state(mo.callout(mo.md("<h3 style='text-align: left;'>⚠️ Please enter an asking price.</h3>"), kind='warn'))
    return (calling_decision,)


@app.cell
def _(calling_decision, mo):
    decision_button = mo.ui.button(on_click=calling_decision, label="Make Decision")
    return (decision_button,)


@app.cell
def _(mo):
    get_decision_state, set_decision_state = mo.state(mo.callout(mo.md("<h3 style='text-align: left;'>Fill in the house details above and click Make Decision.</h3>"), kind='info'))
    return get_decision_state, set_decision_state


@app.cell
def _(
    Housing,
    Session,
    engine,
    first_flr_sf,
    full_bath,
    garage_area,
    gr_liv_area,
    lot_area,
    overall_cond,
    overall_qual,
    pid,
    render_table,
    sale_price,
    select,
    set_table_state,
    total_bsmt_sf,
    year_built,
    year_remod_add,
):
    def delete_house(*_):
        with Session(engine) as session:
            statement = select(Housing).where(Housing.pid == int(pid.value))
            results = session.exec(statement)  
            the_house = results.one()   

            session.delete(the_house)  
            session.commit()  

            set_table_state(render_table())

    def add_house(*_):
        my_dict = {'pid': pid.value,
                   "overall_qual": overall_qual.value,
                   "gr_liv_area": gr_liv_area.value, 
                   "first_flr_sf": first_flr_sf.value, 
                   "year_built": year_built.value, 
                   "year_remod_add": year_remod_add.value, 
                   "lot_area": lot_area.value, 
                   "overall_cond": overall_cond.value, 
                   "garage_area": garage_area.value, 
                   "total_bsmt_sf": total_bsmt_sf.value, 
                   "full_bath": full_bath.value, 
                   "sale_price": sale_price.value }

        converted_dict = {k: int(v) for k, v in my_dict.items() if v} # Only include non-empty inputs

        with Session(engine) as session:
            new_house = Housing(**converted_dict)
            session.add(new_house)
            session.commit()
            set_table_state(render_table())


    def update_house(*_):
        features = ['pid', "overall_qual", "gr_liv_area",
                    "first_flr_sf", "year_built", "year_remod_add",
                    "lot_area", "overall_cond", "garage_area", "total_bsmt_sf",
                    "full_bath", "sale_price"]

        widgets_mapping = {
            "pid": pid,
            "overall_qual": overall_qual,
            "gr_liv_area": gr_liv_area,
            "first_flr_sf": first_flr_sf,
            "year_built": year_built,
            "year_remod_add": year_remod_add,
            "lot_area": lot_area,
            "overall_cond": overall_cond,
            "garage_area": garage_area,
            "total_bsmt_sf": total_bsmt_sf,
            "full_bath": full_bath,
            "sale_price": sale_price,
        }


        with Session(engine) as session:
            statement = select(Housing).where(Housing.pid == int(pid.value))
            result = session.exec(statement)
            the_house = result.one()

            for f in features:
                setattr(the_house, f, int(widgets_mapping[f].value))

            session.add(the_house)
            session.commit()
            set_table_state(render_table())
    return add_house, delete_house, update_house


@app.function
def json_housings(housings):
    return [house.model_dump() for house in housings]


@app.cell
def _(Housing, Session, engine, mo, select):
    def render_table():
            with Session(engine) as session:
                try:
                    houses = session.exec(select(Housing)).all()
                except:
                    return mo.md("No data available!")
            return json_housings(houses)
    return (render_table,)


@app.cell
def _(mo, render_table):
    table_state, set_table_state = mo.state(render_table())
    return set_table_state, table_state


@app.cell
def _(mo, table_state):
    crud_table = mo.ui.table(data=table_state(), selection='single',initial_selection=[0],show_download=False)
    return (crud_table,)


@app.cell
def _(crud_table, mo):
    pid = mo.ui.text(value=str(crud_table.value[0]['pid']) if crud_table.value else "", label='**PID:**', placeholder="Input Data...")

    overall_qual = mo.ui.text(value=str(crud_table.value[0]['overall_qual']) if crud_table.value else "", label='**Overall Quality of House:**', placeholder="Input Data...")

    gr_liv_area = mo.ui.text(value=str(crud_table.value[0]['gr_liv_area']) if crud_table.value else "", label='**Ground Living Area Square Footage:**', placeholder="Input Data...")

    first_flr_sf = mo.ui.text(value=str(crud_table.value[0]['first_flr_sf']) if crud_table.value else "", label='**First Floor Square Footage:**', placeholder="Input Data...")

    year_built = mo.ui.text(value=str(crud_table.value[0]['year_built']) if crud_table.value else "", label='**Year Built:**', placeholder="Input Data...")

    year_remod_add = mo.ui.text(value=str(crud_table.value[0]['year_remod_add']) if crud_table.value else "", label='**Year Remodeled:**', placeholder="Input Data...")

    lot_area = mo.ui.text(value=str(crud_table.value[0]['lot_area']) if crud_table.value else "", label='**Lot Area:**', placeholder="Input Data...")

    overall_cond = mo.ui.text(value=str(crud_table.value[0]['overall_cond']) if crud_table.value else "", label='**Overall Condition of House:**', placeholder="Input Data...")

    garage_area = mo.ui.text(value=str(crud_table.value[0]['garage_area']) if crud_table.value else "", label='**Garage Area Square Footage:**', placeholder="Input Data...")

    total_bsmt_sf = mo.ui.text(value=str(crud_table.value[0]['total_bsmt_sf']) if crud_table.value else "", label='**Total Basement Square Footage:**', placeholder="Input Data...")

    full_bath = mo.ui.text(value=str(crud_table.value[0]['full_bath']) if crud_table.value else "", label='**Number of Bathrooms:**', placeholder="Input Data...")

    sale_price = mo.ui.text(value=str(crud_table.value[0]['sale_price']) if crud_table.value else "", label='**Sale Price:**', placeholder="Input Data...")
    return (
        first_flr_sf,
        full_bath,
        garage_area,
        gr_liv_area,
        lot_area,
        overall_cond,
        overall_qual,
        pid,
        sale_price,
        total_bsmt_sf,
        year_built,
        year_remod_add,
    )


@app.cell
def _(
    first_flr_sf,
    full_bath,
    garage_area,
    gr_liv_area,
    lot_area,
    mo,
    overall_cond,
    overall_qual,
    pid,
    sale_price,
    total_bsmt_sf,
    year_built,
    year_remod_add,
):
    card = mo.callout(mo.hstack([mo.vstack([pid, overall_qual, gr_liv_area, first_flr_sf], gap=2),
                                mo.vstack([year_built, year_remod_add, lot_area, overall_cond], gap=2), 
                                mo.vstack([garage_area, total_bsmt_sf, full_bath, sale_price], gap=2)
                                ], gap=2), kind='info')
    return (card,)


@app.cell
def _(add_house, delete_house, mo, update_house):
    update_button = mo.ui.button(label="Update House", on_click=update_house)
    delete_button = mo.ui.button(label="Delete House", on_click=delete_house)
    add_button = mo.ui.button(label="Add House", on_click=add_house)
    return add_button, delete_button, update_button


@app.cell
def _(add_button, card, crud_table, delete_button, mo, update_button):
    def crud_page():
        return mo.vstack([
            mo.md("<h1 style='margin-bottom: 10px; text-align: left; color: green;'>Housing Database</h1>"),
            mo.md("Historical house records from the Ames, Iowa dataset used to train the model."),
            crud_table,
            mo.vstack([card, mo.hstack([update_button, delete_button, add_button], justify='center', gap=4)], gap=2),
        ], gap=1)
    return (crud_page,)


@app.cell
def _(
    asking_price,
    decision_button,
    get_decision_state,
    input_first_flr_sf,
    input_full_bath,
    input_garage_area,
    input_gr_liv_area,
    input_lot_area,
    input_overall_cond,
    input_overall_qual,
    input_total_bsmt_sf,
    input_year_built,
    input_year_remod,
    mo,
):
    def home_page():
        form = mo.callout(
            mo.vstack([
                asking_price,
                mo.md("---"),
                mo.md("**House Attributes** — leave unknown fields at their default (training average)"),
                mo.hstack([
                    mo.vstack([input_overall_qual, input_gr_liv_area, input_first_flr_sf, input_year_built, input_year_remod], gap=3),
                    mo.vstack([input_overall_cond, input_full_bath, input_lot_area, input_garage_area, input_total_bsmt_sf], gap=3),
                ], gap=6),
            ], gap=2),
            kind='info'
        )
        return mo.vstack([
            mo.md("<h1 style='margin-bottom: 10px; text-align: left; color: green;'>Main Page</h1>"),
            mo.md("⚠️ *This model was trained on Ames, Iowa housing data. Predictions are only reliable right now for houses with similar characteristics and price ranges (~$100k–$400k).*"),
            form,
            decision_button,
            get_decision_state(),
        ], gap=2)
    return (home_page,)


@app.cell
def _(mo):
    def about_page():
        return mo.vstack([mo.md("""<h1 style='margin-bottom: 30px; text-align: left; color: green;'>About Page</h1>"""), 
                          mo.hstack([mo.vstack([mo.md("This app is designed to assist users in making house-purchasing decisions. Users can browse available house data from the database, input an asking price, and receive purchase advice generated by the app. A key feature is that the advice reflects the model's degree of confidence, explicitly quantifying uncertainty. This is achieved through training a **Bayesian Regression model**, which generates posterior distributions that inform the decision-making process."),
                          mo.md("The process began with raw data from Kaggle's Ames Housing Dataset (CSV). This data was cleaned and transformed using **SQLMesh** and the raw CSV data was ingested and incrementally transformed through a pipeline of SQLMesh models, where each model built on the outputs of the previous stage, resulting in the final cleaned dataset. Data cleaning and transformations include handling outliers, taking the logarithm of features, and standardization. The final cleaned dataset was queried into a Pandas DataFrame and used to train the Bayesian regression model via **stochastic variational inference (SVI)**. This was implemented using **Pyro**, a probabilistic programming language, which allowed us to define the model's joint probability density (the product of priors and likelihood) and specify a guide—Pyro's term for the variational distribution that approximates the true posterior. Approximating the posterior is necessary because computing the evidence in Bayes' rule is typically intractable. To perform the approximation, SVI optimizes an objective function known as the **Evidence Lower Bound (ELBO)**. The algorithm takes stochastic gradient steps to maximize the ELBO, which is mathematically equivalent to minimizing the **Kullback–Leibler (KL) divergence (a measure of difference in information between two distributions or relative entropy)** between the variational distribution and the true posterior. Intuitively, a smaller KL divergence means the approximation is closer to the true posterior, with a value of zero indicating an exact match. After the model's posterior distributions were computed, credible intervals were derived and used in the decision logic. Finally, the user interface was built with **marimo (reactive Python notebook)**, and CRUD functionality was implemented with **SQLModel (Object-relational mapping)**."),
        mo.md("Most of the model's credible intervals indicate roughly 46% uncertainty between the lower and upper bounds. This corresponds to approximately ±23% uncertainty on either side of the mean prediction. In other words, for a typical prediction, the model is 90% confident (because 90% credible interval) that the true sale price lies within ±23% of its predicted value.")
        ]),
              mo.vstack([mo.md("For the decision-making component of this app, the model uses the width of the credible intervals. Specifically, if the asking price is below the lower bound of the interval, the house is considered a bargain, since the lowest price predicted by the model (the lower bound) exceeds the asking price. Conversely, if the asking price is above the upper bound, the house is considered overpriced. If the asking price falls within the interval, the difference between the mean of the predicted sale prices and the asking price determines potential gain or loss: positive values indicate a potential gain, while negative values indicate a potential loss.")
                         ,mo.md("In future versions, potential improvements include adding more ways for users to interact with the data beyond the scatterplot and eliminating the shuttering that occurs each time an action is performed. Additionally, the app could be enhanced by integrating the CRUD functionality with the house decision logic so that any CRUD actions are immediately reflected in the decision-making output.")])                       
                                               ]),
                          mo.vstack([mo.image(src="images/Predicted_vs._True.png", width=650, height=500, rounded=True, caption="You can see that the model is pretty accurately predicting the true sale prices."),
                          mo.image(src="images/Model_Performance_on_Test_Set.png", width=900, height=500, rounded=True, caption="Model Performance. Generalizes well on unseen data."),
                           mo.image(src="images/Relative_Widths.png", width=550, height=400, rounded=True, caption="Majority of relative widths are around 46% (so ±23%). Distribution is Gaussian.")         ], gap=2)
                         ], gap=1)
    return (about_page,)


@app.cell
def _(about_page, crud_page, home_page, mo):
    mo.routes(
        {
            "#/": mo.Html(f"{home_page()}"),
            "#/CRUD": mo.Html(f"{crud_page()}"),
            "#/about": mo.Html(f"{about_page()}"),
            mo.routes.CATCH_ALL: mo.Html(f"{home_page()}")
        }
    )
    return


@app.cell
def _(mo):
    mo.sidebar([
        mo.md("""<h1 style='margin-bottom: 25px;'>House Buying Application</h1>"""),
        mo.nav_menu(
            {
                "#/": f"{mo.icon('lucide:home', color='purple')} Home",
                "#/CRUD": f"{mo.icon('icon-park:data-all')} Housing Database",
                "#/about": f"{mo.icon('unjs:unctx')} About Me",
            },
            orientation="vertical",
        ),
    ])
    return



if __name__ == "__main__":
    app.run()
