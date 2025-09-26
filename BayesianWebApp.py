import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import pandas as pd
    import os
    import pyro
    import torch
    import seaborn as sns
    import pyro.distributions as dist
    import pyro.distributions.constraints as constraints
    import logging
    import matplotlib.pyplot as plt 
    import numpy as np
    import random
    import plotly.express as px
    import plotly.graph_objects as go
    from sqlmodel import Field, Session, SQLModel, create_engine, select, func
    from typing import Optional
    return (
        Field,
        Optional,
        SQLModel,
        Session,
        create_engine,
        dist,
        duckdb,
        logging,
        mo,
        np,
        os,
        pd,
        plt,
        px,
        pyro,
        select,
        torch,
    )




@app.cell
def _(duckdb):
    con = duckdb.connect("sqlmesh/houses.db")
    return (con,)


@app.cell
def _(con):
    df = con.execute("SELECT * FROM sqlmesh.standardized_model").fetchdf()
    return (df,)


@app.cell
def _(con):
    df_drop_lot_area = con.execute("SELECT * FROM sqlmesh.drop_lot_area").fetchdf()
    return (df_drop_lot_area,)


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
def _(df, torch):
    train = torch.tensor(df.values, dtype=torch.float)
    gr_liv_area_std, first_flr_sf_std, lot_area_std, garage_area_std, total_bsmt_sf_std, qual_livarea_interaction_std, overall_qual_1, overall_qual_2, overall_qual_3, overall_qual_4, overall_qual_5, overall_qual_6, overall_qual_7, overall_qual_8, overall_qual_9, overall_qual_10, year_built_std, year_remod_add_std, overall_cond_1, overall_cond_2, overall_cond_3, overall_cond_4, overall_cond_5, overall_cond_6, overall_cond_7, overall_cond_8, overall_cond_9, full_bath_0, full_bath_1, full_bath_2, full_bath_3, full_bath_4, sale_price_log_std = train[:, 1], train[:, 2], train[:, 3], train[:, 4], train[:, 5], train[:, 6], train[:, 7], train[:, 8], train[:, 9], train[:, 10], train[:, 11], train[:, 12], train[:, 13], train[:, 14], train[:, 15], train[:, 16], train[:, 17], train[:, 18], train[:, 19], train[:, 20], train[:, 21], train[:, 22], train[:, 23], train[:, 24], train[:, 25], train[:, 26], train[:, 27], train[:, 28], train[:, 29], train[:, 30], train[:, 31], train[:, 32], train[:, 33]

    target_std = 0.4073182750630981 # standard deviation of sale_price_log
    target_mean = 12.020431074250771 # mean of sale_price_log. Cool to know. I basically used this to reverse the standardization
    return (
        first_flr_sf_std,
        full_bath_0,
        full_bath_1,
        full_bath_2,
        full_bath_3,
        full_bath_4,
        garage_area_std,
        gr_liv_area_std,
        lot_area_std,
        overall_cond_1,
        overall_cond_2,
        overall_cond_3,
        overall_cond_4,
        overall_cond_5,
        overall_cond_6,
        overall_cond_7,
        overall_cond_8,
        overall_cond_9,
        overall_qual_1,
        overall_qual_10,
        overall_qual_2,
        overall_qual_3,
        overall_qual_4,
        overall_qual_5,
        overall_qual_6,
        overall_qual_7,
        overall_qual_8,
        overall_qual_9,
        qual_livarea_interaction_std,
        sale_price_log_std,
        target_mean,
        target_std,
        total_bsmt_sf_std,
        year_built_std,
        year_remod_add_std,
    )


@app.cell
def _(dist, pyro):
    def model(gr_liv_area_std, first_flr_sf_std, lot_area_std, garage_area_std, total_bsmt_sf_std, qual_livarea_interaction_std, overall_qual_1, overall_qual_2, overall_qual_3, overall_qual_4, overall_qual_5, overall_qual_6, overall_qual_7, overall_qual_8, overall_qual_9, overall_qual_10, year_built_std, year_remod_add_std, overall_cond_1, overall_cond_2, overall_cond_3, overall_cond_4, overall_cond_5, overall_cond_6, overall_cond_7, overall_cond_8, overall_cond_9, full_bath_0, full_bath_1, full_bath_2, full_bath_3, full_bath_4, sale_price_log_std=None):
        alpha = pyro.sample("alpha", dist.Normal(0, 5))
        b1 = pyro.sample("b1", dist.Normal(0, 1))
        b2 = pyro.sample("b2", dist.Normal(0, 1))
        b3 = pyro.sample("b3", dist.Normal(0, 1))
        b4 = pyro.sample("b4", dist.Normal(0, 1))
        b5 = pyro.sample("b5", dist.Normal(0, 1))
        b6 = pyro.sample("b6", dist.Normal(0, 1))

        ### Overall Quality features
        b7 = pyro.sample("b7", dist.Normal(0, 0.5))
        b8 = pyro.sample("b8", dist.Normal(0, 0.5))
        b9 = pyro.sample("b9", dist.Normal(0, 0.5))
        b10 = pyro.sample("b10", dist.Normal(0, 0.5))
        b11 = pyro.sample("b11", dist.Normal(0, 0.5))
        b12 = pyro.sample("b12", dist.Normal(0, 0.5))
        b13 = pyro.sample("b13", dist.Normal(0, 0.5))
        b14 = pyro.sample("b14", dist.Normal(0, 0.5))
        b15 = pyro.sample("b15", dist.Normal(0, 0.5))
        b16 = pyro.sample("b16", dist.Normal(0, 0.5))


        b17 = pyro.sample("b17", dist.Normal(0, 1))
        b18 = pyro.sample("b18", dist.Normal(0, 1))

        ### Overall Condition features
        b19 = pyro.sample("b19", dist.Normal(0, 0.5))
        b20 = pyro.sample("b20", dist.Normal(0, 0.5))
        b21 = pyro.sample("b21", dist.Normal(0, 0.5))
        b22 = pyro.sample("b22", dist.Normal(0, 0.5))
        b23 = pyro.sample("b23", dist.Normal(0, 0.5))
        b24 = pyro.sample("b24", dist.Normal(0, 0.5))
        b25 = pyro.sample("b25", dist.Normal(0, 0.5))
        b26 = pyro.sample("b26", dist.Normal(0, 0.5))
        b27 = pyro.sample("b27", dist.Normal(0, 0.5))

        ### Full Bath Features
        b28 = pyro.sample("b28", dist.Normal(0, 0.5))
        b29 = pyro.sample("b29", dist.Normal(0, 0.5))
        b30 = pyro.sample("b30", dist.Normal(0, 0.5))
        b31 = pyro.sample("b31", dist.Normal(0, 0.5))
        b32 = pyro.sample("b32", dist.Normal(0, 0.5))

        sigma = pyro.sample("sigma", dist.HalfNormal(1))

        mean = alpha + b1 * gr_liv_area_std + b2 * first_flr_sf_std + b3 * lot_area_std + b4 * garage_area_std + b5 * total_bsmt_sf_std + b6 * qual_livarea_interaction_std + b7 * overall_qual_1 + b8 * overall_qual_2 + b9 * overall_qual_3 + b10 * overall_qual_4 + b11 * overall_qual_5 + b12 * overall_qual_6 + b13 * overall_qual_7 + b14 * overall_qual_8 + b15 * overall_qual_9 + b16 * overall_qual_10 + b17 * year_built_std + b18 * year_remod_add_std + b19 * overall_cond_1 + b20 * overall_cond_2 + b21 * overall_cond_3 + b22 * overall_cond_4 + b23 * overall_cond_5 + b24 * overall_cond_6 + b25 * overall_cond_7 + b26 * overall_cond_8 + b27 * overall_cond_9 + b28 * full_bath_0 + b29 * full_bath_1 + b30 * full_bath_2 + b31 * full_bath_3 + b32 * full_bath_4

        with pyro.plate("data", len(gr_liv_area_std)):
            return pyro.sample("obs", dist.Normal(mean, sigma), obs=sale_price_log_std)
    return (model,)


@app.cell
def _(logging, plt, pyro):
    smoke_test = False # Set to True if you want to test/debug the SVI training phase
    assert pyro.__version__.startswith('1.9.1')

    pyro.enable_validation(True)
    pyro.set_rng_seed(1)
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # Set matplotlib settings

    plt.style.use('default')
    return (smoke_test,)


@app.cell
def _(
    first_flr_sf_std,
    full_bath_0,
    full_bath_1,
    full_bath_2,
    full_bath_3,
    full_bath_4,
    garage_area_std,
    gr_liv_area_std,
    lot_area_std,
    mo,
    model,
    overall_cond_1,
    overall_cond_2,
    overall_cond_3,
    overall_cond_4,
    overall_cond_5,
    overall_cond_6,
    overall_cond_7,
    overall_cond_8,
    overall_cond_9,
    overall_qual_1,
    overall_qual_10,
    overall_qual_2,
    overall_qual_3,
    overall_qual_4,
    overall_qual_5,
    overall_qual_6,
    overall_qual_7,
    overall_qual_8,
    overall_qual_9,
    plt,
    pyro,
    qual_livarea_interaction_std,
    sale_price_log_std,
    smoke_test,
    total_bsmt_sf_std,
    year_built_std,
    year_remod_add_std,
):
    pyro.clear_param_store()


    auto_guide = pyro.infer.autoguide.AutoNormal(model)
    adam = pyro.optim.Adam({"lr": 0.0095})
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

    # Training Model
    losses = []
    for step in mo.status.progress_bar(range(2000 if not smoke_test else 2), title="Training Model First", subtitle="Please Wait...", show_eta=True, show_rate=True):
        loss = svi.step(gr_liv_area_std, first_flr_sf_std, lot_area_std, garage_area_std, total_bsmt_sf_std, qual_livarea_interaction_std, overall_qual_1, overall_qual_2, overall_qual_3, overall_qual_4, overall_qual_5, overall_qual_6, overall_qual_7, overall_qual_8, overall_qual_9, overall_qual_10, year_built_std, year_remod_add_std, overall_cond_1, overall_cond_2, overall_cond_3, overall_cond_4, overall_cond_5, overall_cond_6, overall_cond_7, overall_cond_8, overall_cond_9, full_bath_0, full_bath_1, full_bath_2, full_bath_3, full_bath_4, sale_price_log_std)
        losses.append(loss)
        if step % 100 == 0:
            print(f"Step {step}: ELBO loss = {loss}")

    # Plotting ELBO Loss
    fig_svi, ax_svi = plt.subplots(figsize=(5, 2))
    ax_svi.plot(losses)
    ax_svi.set_xlabel("SVI step")
    ax_svi.set_ylabel("ELBO loss")
    plt.show()
    mo.md("") # This is just to make the progress bar disappear once it is done. I don't know how else to make it disappear without rendering anything.
    return (auto_guide,)


@app.cell
def _(
    auto_guide,
    first_flr_sf_std,
    full_bath_0,
    full_bath_1,
    full_bath_2,
    full_bath_3,
    full_bath_4,
    garage_area_std,
    gr_liv_area_std,
    lot_area_std,
    model,
    overall_cond_1,
    overall_cond_2,
    overall_cond_3,
    overall_cond_4,
    overall_cond_5,
    overall_cond_6,
    overall_cond_7,
    overall_cond_8,
    overall_cond_9,
    overall_qual_1,
    overall_qual_10,
    overall_qual_2,
    overall_qual_3,
    overall_qual_4,
    overall_qual_5,
    overall_qual_6,
    overall_qual_7,
    overall_qual_8,
    overall_qual_9,
    pd,
    pyro,
    qual_livarea_interaction_std,
    sale_price_log_std,
    total_bsmt_sf_std,
    year_built_std,
    year_remod_add_std,
):
    predictive = pyro.infer.Predictive(model, guide=auto_guide, num_samples=800)
    svi_samples = predictive(gr_liv_area_std, first_flr_sf_std, lot_area_std, garage_area_std, total_bsmt_sf_std, qual_livarea_interaction_std, overall_qual_1, overall_qual_2, overall_qual_3, overall_qual_4, overall_qual_5, overall_qual_6, overall_qual_7, overall_qual_8, overall_qual_9, overall_qual_10, year_built_std, year_remod_add_std, overall_cond_1, overall_cond_2, overall_cond_3, overall_cond_4, overall_cond_5, overall_cond_6, overall_cond_7, overall_cond_8, overall_cond_9, full_bath_0, full_bath_1, full_bath_2, full_bath_3, full_bath_4, sale_price_log_std=None)
    svi_sale_price_log = svi_samples["obs"]

    y_mean = svi_sale_price_log.mean(0).detach().cpu().numpy()
    y_perc_5 = svi_sale_price_log.kthvalue(int(0.05 * len(svi_sale_price_log)), dim=0)[0].detach().cpu().numpy()
    y_perc_95 = svi_sale_price_log.kthvalue(int(0.95 * len(svi_sale_price_log)), dim=0)[0].detach().cpu().numpy()
    sale_price_log_std_numpy = sale_price_log_std.detach().cpu().numpy()

    predictions = pd.DataFrame({
        "y_mean": y_mean,
        "y_perc_5": y_perc_5,
        "y_perc_95": y_perc_95,
        "true_y": sale_price_log_std_numpy,
    })
    return (predictions,)


@app.cell
def _(df_drop_lot_area, np, predictions, target_mean, target_std):
    #target_std_np = target_std.numpy() if isinstance(target_std, torch.Tensor) else target_std
    #target_mean_np = target_mean.numpy() if isinstance(target_mean, torch.Tensor) else target_mean

    # Un-standardize
    predictions["true_y_orig"] = predictions["true_y"] * target_std + target_mean
    predictions["y_mean_orig"] = predictions["y_mean"] * target_std + target_mean
    predictions["y_perc_5_orig"] = predictions["y_perc_5"] * target_std + target_mean
    predictions["y_perc_95_orig"] = predictions["y_perc_95"] * target_std + target_mean

    # Undo log to get actual sale price
    predictions["TrueSalePrice"] = np.exp(predictions["true_y_orig"])
    predictions["PredictedSalePrice"] = np.exp(predictions["y_mean_orig"])
    predictions["LowerBoundCI"] = np.exp(predictions["y_perc_5_orig"])
    predictions["UpperBoundCI"] = np.exp(predictions["y_perc_95_orig"])

    predictions['HouseID'] = np.arange(len(predictions))

    predictions["GroundLivingArea"] = df_drop_lot_area['gr_liv_area']
    predictions["BasementSquareFootage"] = df_drop_lot_area['total_bsmt_sf']
    predictions["OverallQuality"] = df_drop_lot_area['overall_qual']
    return


@app.cell
def _(mo, predictions, px):
    scatter_fig = px.scatter(predictions,  x='HouseID', y='PredictedSalePrice', title="House Data", subtitle = "Click on a Data Point for More Information about a House", hover_data=['LowerBoundCI', 'UpperBoundCI', 'GroundLivingArea', 'BasementSquareFootage', 'OverallQuality'])


    scatter_fig.update_layout(clickmode="event+select")

    scatter_fig.update_layout(
        xaxis_title="House ID",   # user sees this
        yaxis_title="Mean Predicted Sale Price"
    )

    scatter_fig.update_layout(dragmode="pan")

    scatter_fig.update_traces(
        selected=dict(marker=dict(color="red", size=12)),
        unselected=dict(marker=dict(opacity=0.2))
    ) # Turns red when you select a point

    scatter_fig.update_layout(
        xaxis=dict(range=[-100, 3100], fixedrange=True),   # don’t let user zoom out further than this
        yaxis=dict(range=[min(predictions['PredictedSalePrice']) - 400, max(predictions['PredictedSalePrice']) + 400
                         ])
    )

    reactive_chart = mo.ui.plotly(scatter_fig, config={
            "modeBarButtonsToRemove": ["select2d", "lasso2d"]
    })
    return (reactive_chart,)


@app.cell
def _(mo):
    def decision(y_low, y_high, predicted_mean, asking_price, tolerance=0.025, margin_threshold=2000): # This function is for a single house (i.e. one row of data)
        interval_width = y_high - y_low
        relative_width = ((interval_width / predicted_mean) * 100) // 2
        expected_margin = predicted_mean - asking_price

        # tolerance is % wiggle room you’re comfortable with
        if y_high < asking_price * (1 - tolerance):
            return mo.callout(mo.md("<h3 style='text-align: left;'>🛑 Don't Buy (overpriced). The asking price is a lot higher than what this house would realistically sell for.</h3>"), kind="danger")
        elif y_low > asking_price * (1 + tolerance):
            return mo.callout(mo.md("<h3 style='text-align: left;'>✅ Buy (undervalued). The asking price is below some of the lowest prices that this house would realistically sell for.</h3>"), kind="success")
        else:
            if expected_margin > margin_threshold:
                return mo.callout(mo.md(f"<h3 style='text-align: left;'>✅ Buy. Expected gain if you buy: ${expected_margin}. I'm 90% confident that this is the case.</h3>"), kind="success")
            elif expected_margin < -margin_threshold:
                return mo.callout(mo.md(f"<h3 style='text-align: left;'>🛑 Don't Buy. Expected loss if you buy: ${expected_margin}. I'm 90% confident that purchasing this house is not worth it.</h3>"), kind="danger")
            else:
                return mo.callout(mo.md(f"<h3 style='text-align: left;'>🤔 The asking price falls within a reasonable range. Expected margin: ${expected_margin}. There is a 90% probability that this represents a fair purchase."), kind="warn")
    return (decision,)


@app.cell
def _(asking_price, decision, mo, reactive_chart, set_decision_state):
    lower_bound_plot = reactive_chart.value[0]['LowerBoundCI'] if reactive_chart.value else None
    upper_bound_plot = reactive_chart.value[0]['UpperBoundCI'] if reactive_chart.value else None
    predicted_sale_price_plot = reactive_chart.value[0]['PredictedSalePrice'] if reactive_chart.value else None

    def calling_decision(*_):
        if lower_bound_plot and upper_bound_plot and predicted_sale_price_plot and asking_price.value:
            set_decision_state(decision(lower_bound_plot, upper_bound_plot, predicted_sale_price_plot, asking_price.value))
        else:
            set_decision_state(mo.callout(mo.md("""<h3 style='text-align: left;'>⚠️ One or more missing inputs!</h3>"""), kind='warn'))
    return (calling_decision,)


@app.cell
def _(mo):
    asking_price = mo.ui.number(start=0, stop=10000000, label="Input an Asking Price for House")
    return (asking_price,)


@app.cell
def _(calling_decision, mo):
    decision_button = mo.ui.button(on_click=calling_decision, label="Make Decision")
    return (decision_button,)


@app.cell
def _(mo):
    get_decision_state, set_decision_state = mo.state(mo.callout(mo.md("<h3 style='text-align: left;'>Waiting for a decision to be made...</h3>"), kind='info'))
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
        return mo.vstack([mo.md("""<h1 style='margin-bottom: 30px; text-align: left; color: green;'>Interact with Database</h1>"""), crud_table, mo.vstack([card, mo.hstack([update_button, delete_button, add_button], justify='center', gap=4)], gap=2)
                         ]
                         , gap=1)
    return (crud_page,)


@app.cell
def _(asking_price, decision_button, get_decision_state, mo, reactive_chart):
    def home_page():
        return mo.vstack(
            [mo.vstack([mo.md("""<h1 style='margin-bottom: 30px; text-align: left; color: green;'>Main Page</h1>"""), reactive_chart, mo.ui.table(reactive_chart.value, show_download=False)], gap=0.001),  
                          mo.vstack([asking_price, decision_button], gap=2), 
                          get_decision_state()
                         ], 
            gap=1.5)
    return (home_page,)


@app.cell
def _(mo):
    def about_page():
        return mo.vstack([mo.md("""<h1 style='margin-bottom: 30px; text-align: left; color: green;'>About Page</h1>"""), 
                          mo.hstack([mo.vstack([mo.md("This app is designed to assist users in making house-purchasing decisions. Users can browse available house data from the database, input an asking price, and receive purchase advice generated by the app. A key feature is that the advice reflects the model’s degree of confidence, explicitly quantifying uncertainty. This is achieved through training a **Bayesian Regression model**, which generates posterior distributions that inform the decision-making process."),
                          mo.md("The process began with raw data from Kaggle’s Ames Housing Dataset (CSV). This data was cleaned and transformed using **SQLMesh** and the raw CSV data was ingested and incrementally transformed through a pipeline of SQLMesh models, where each model built on the outputs of the previous stage, resulting in the final cleaned dataset. Data cleaning and transformations include handling outliers, taking the logarithm of features, and standardization. The final cleaned dataset was queried into a Pandas DataFrame and used to train the Bayesian regression model via **stochastic variational inference (SVI)**. This was implemented using **Pyro**, a probabilistic programming language, which allowed us to define the model’s joint probability density (the product of priors and likelihood) and specify a guide—Pyro’s term for the variational distribution that approximates the true posterior. Approximating the posterior is necessary because computing the evidence in Bayes’ rule is typically intractable. To perform the approximation, SVI optimizes an objective function known as the **Evidence Lower Bound (ELBO)**. The algorithm takes stochastic gradient steps to maximize the ELBO, which is mathematically equivalent to minimizing the **Kullback–Leibler (KL) divergence (a measure of difference in information between two distributions or relative entropy)** between the variational distribution and the true posterior. Intuitively, a smaller KL divergence means the approximation is closer to the true posterior, with a value of zero indicating an exact match. After the model’s posterior distributions were computed, credible intervals were derived and used in the decision logic. Finally, the user interface was built with **marimo (reactive Python notebook)**, and CRUD functionality was implemented with **SQLModel (Object-relational mapping)**."),
        mo.md('Most of the model’s credible intervals indicate roughly 46% uncertainty between the lower and upper bounds. This corresponds to approximately ±23% uncertainty on either side of the mean prediction. In other words, for a typical prediction, the model is 90% confident (because 90% credible interval) that the true sale price lies within ±23% of its predicted value.')                            
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
            "#/": home_page,
            "#/CRUD": crud_page,
            "#/about": about_page,
            mo.routes.CATCH_ALL: home_page,
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
                "#/CRUD": f"{mo.icon('icon-park:data-all')} Access CRUD Functionality",
                "#/about": f"{mo.icon('unjs:unctx')} About Me",
            },
            orientation="vertical",
        ),
    ])
    return


@app.cell
def _():
    ### Notes:

    # How to intepret credible intervals: based on the percentage of the credible interval (say 90% credible interval) and its width, the correct interpretation is that the model is 90% certain that the true target variable is within the interval width. Narrower widths tell you that the model is more certain that the true target variable is within a smaller range of values, and how certain depends on the percentage of the credible interval (or how much probability mass the interval covers).

    ### Most of the intervals from this model are around 44 percent uncertainty for the total lower and upper bounds. 
    # So the model is giving around 22% uncertainty on either side of the mean prediction.
    ### So that means:
    # For a typical prediction, the model is 90% certain (because 90 percent credible interval) that the true value lies within ±22% of the predicted price. 
    # In other words, the model is 90 percent confident that the true sale price is within ±22% of its prediction.

    # For the decision part of this app, the app uses the width of the credible intervals. More specifically, if the asking price is below the lower bound of the interval, it is a bargain according to the model because the cheapest price the model predicts (i.e., the lower bound) is greater than the asking price. This applies vice-versa as well. If the asking price is within the model's interval, the difference between the predicted mean and the asking price will determine gain or loss. Positive results would indicate gain, whereas negative results would indicate loss.

    # Try to include all the plots made during model development to showcase model performance etc., kind of like an appendix. For example, showing the plot of how the model performs on the test set.

    # For future commits, improvements include making more ways users can interact with the data besides a scatterplot and eliminating the shuttering in the app everytime an action is performed. Additionally, the app can be improved by connecting the CRUD functionality to the house decision making so that any CRUD actions will be reflected.
    return


if __name__ == "__main__":
    app.run()
