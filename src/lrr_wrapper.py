# ***********************************************************************
# * Licensed Materials - Property of IBM
# *
# * IBM SPSS Products: Statistics Common
# *
# * (C) Copyright IBM Corp. 1989, 2022
# *
# * US Government Users Restricted Rights - Use, duplication or disclosure
# * restricted by GSA ADP Schedule Contract with IBM Corp.
# ************************************************************************

from wrapper.basewrapper import *
from wrapper import wraputil
from util.statjson import *

import numpy as np
import warnings
import traceback

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

warnings.simplefilter("error", category=RuntimeWarning)

"""Initialize the lrr wrapper"""
init_wrapper("lrr", os.path.join(os.path.dirname(__file__), "LRR-properties.json"))

standardize = intercept = True
holdout = False
trace_table_max_rows = 0
cycle_patterns = False


def execute(iterator_id, data_model, settings, lang="en"):
    fields = data_model["fields"]
    output_json = StatJSON(get_name())

    intl = get_lang_resource(lang)

    def execute_model(data):
        if data is not None:
            case_count = len(data)
        else:
            return

        try:
            check_settings(settings, fields)
            global standardize, intercept, holdout, trace_table_max_rows, cycle_patterns
            standardize = get_value("criteria_standardize")
            intercept = get_value("criteria_intercept")
            holdout_value = get_value("partition_holdout") if is_set("partition_holdout") else 0.0
            holdout = True if holdout_value > 0.0 else False
            trace_table_max_rows = get_value("criteria_trace")
            if trace_table_max_rows is None:
                trace_table_max_rows = 0
            cycle_patterns = get_output_value("patterns")
            if cycle_patterns is None:
                cycle_patterns = False

            metric = get_value("alpha_metric") if is_set("alpha_metric") else None
            alphas = get_value("alpha_values")
            if metric == "LG10":
                for i in range(len(alphas)):
                    alphas[i] = 10 ** alphas[i]

            records = RecordData(data)
            w_name = ""
            for field in fields:
                records.add_type(field["type"])
                if field["metadata"]["modeling_role"] == "frequency":
                    w_name = field["name"]

            y_name = get_value("dependent")

            x_names = []
            x_fnotes = []
            if is_set("factors"):
                x_names.extend(get_value("factors"))
                if is_set("original_factors"):
                    x_fnotes.extend(get_value("original_factors"))
                else:
                    x_fnotes.extend(get_value("factors"))
            if is_set("covariates"):
                x_names.extend(get_value("covariates"))
                x_fnotes.extend(get_value("covariates"))

            columns_data = records.get_columns()
            y_index = wraputil.get_index(fields, y_name)
            w_index = wraputil.get_index(fields, w_name)
            if w_index >= 0:
                if y_index < w_index:
                    w, y = columns_data.pop(w_index), columns_data.pop(y_index)
                else:
                    y, w = columns_data.pop(y_index), columns_data.pop(w_index)
            else:
                y = columns_data.pop(y_index)
                w = None

            partition_name = get_value("partition_variable")
            partition_index = wraputil.get_index(fields, partition_name)
            if partition_index >= 0:
                partition = columns_data.pop()
                if holdout:
                    holdout_indices = [i for i, val in enumerate(partition) if val == 3.0]
                else:
                    holdout_indices = None
                training_indices = [i for i, val in enumerate(partition) if val == 1.0]
            else:
                holdout_indices = None
                training_indices = None

            x = list(zip(*columns_data))

            result = {}

            if isinstance(w, (list, tuple)):
                w = np.array(w)

            x_array = np.array(x)
            y_array = np.array(y)
            mode = get_value("mode").upper()
            if mode == "FIT":
                process_fit(x_array, y_array, w, alphas[0], training_indices, holdout_indices, result)

                if is_set("save_pred") or is_set("save_resid"):
                    pred = get_value("save_pred") if is_set("save_pred") else None
                    resid = get_value("save_resid") if is_set("save_resid") else None
                    save_data(pred, resid, result["pred_values"], result["resid_values"], case_count)

                create_fit_output(y, x_names, x_fnotes, y_name, alphas, intl, output_json, result)
            elif mode == "TRACE":
                process_trace(x_array, y_array, w, alphas, training_indices, result)

                create_trace_output(x_names, x_fnotes, y_name, alphas, intl, output_json, result)
            elif mode == "CROSSVALID":
                nfolds = get_value("criteria_nfolds")
                state = get_value("criteria_state")
                process_cv(x_array, y_array, w, nfolds, state, alphas, training_indices, holdout_indices, result)

                if is_set("save_pred") or is_set("save_resid"):
                    pred = get_value("save_pred") if is_set("save_pred") else None
                    resid = get_value("save_resid") if is_set("save_resid") else None
                    save_data(pred, resid, result["pred_values"], result["resid_values"], case_count)

                create_cv_output(y, x_names, x_fnotes, y_name, nfolds, alphas, intl, output_json, result)

        except Exception as err:
            warning_item = Warnings(intl.loadstring("python_returned_msg") + "\n" + repr(err))
            output_json.add_warnings(warning_item)
            tb = traceback.format_exc()

            notes = Notes(intl.loadstring("python_output"), tb)
            output_json.add_notes(notes)
        finally:
            generate_output(output_json.get_json(), None)
            finish()

    get_records(iterator_id, data_model, execute_model)
    return 0


# process for fit mode
# x: matrix of factors and covariates data which pass by Stats backend
# y: dependent variable data which pass by Stats backend
# swt: weight variable data which pass by Stats backend, it is None if no weight variable
# alphas: single alpha value, For example: 1.0
# t_indices: index list if partition=1
# h_indices: index list if partition=3
# result: A map to save the calculation results
def process_fit(x, y, swt, alphas, t_indices, h_indices, result):
    scaler = int_out = None
    # Get x_train, y_train and swt_train from the original data based on t_indices
    x_train, y_train, swt_train = get_train_group(x, y, swt, t_indices)

    # Get copy of original x, possibly to be scaled
    x_scaled = x

    # Calculate the mean, std, etc base on the template
    if standardize:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train, sample_weight=swt_train)
        x_scaled = scaler.transform(x_scaled)

    linear_model1 = Ridge(alpha=alphas, fit_intercept=intercept, max_iter=100000)
    linear_model1.fit(x_train, y_train, swt_train)

    r2_train = linear_model1.score(x_train, y_train, swt_train)
    result["r2_train"] = r2_train
    bout = linear_model1.coef_
    result["bout"] = bout

    if intercept:
        int_out = linear_model1.intercept_
        result["int_out"] = int_out

    if standardize:
        mean_out = scaler.mean_
        std_out = np.sqrt(scaler.var_)
        raw_bout = bout / scaler.scale_
        result["mean_out"] = mean_out
        result["std_out"] = std_out
        result["raw_bout"] = raw_bout
        if intercept:
            raw_int_out = int_out - np.dot(raw_bout, mean_out)
            result["raw_int_out"] = raw_int_out

    train_pred = linear_model1.predict(x_train)
    train_resid = y_train - train_pred

    result["train_pred"] = train_pred.tolist()
    result["train_resid"] = train_resid.tolist()

    # Get x_holdout, y_holdout and hswt from the original data based on h_indices
    if holdout:
        x_holdout = x[h_indices]
        y_holdout = y[h_indices]
        hswt = None if swt is None else swt[h_indices]

        if standardize:
            x_holdout = scaler.transform(x_holdout)
        r2_holdout = linear_model1.score(x_holdout, y_holdout, hswt)
        result["r2_holdout"] = r2_holdout

        holdout_pred = linear_model1.predict(x_holdout)
        holdout_resid = y_holdout - holdout_pred

        result["holdout_pred"] = holdout_pred.tolist()
        result["holdout_resid"] = holdout_resid.tolist()

    # Calculate the save predicted and residuals value using the original x and y data
    pred = linear_model1.predict(x_scaled)
    resid = y - pred

    result["pred_values"] = pred.tolist()
    result["resid_values"] = resid.tolist()


def process_trace(x, y, swt, alphas, t_indices, result):
    from sklearn.metrics import mean_squared_error
    x_train, y_train, swt_train = get_train_group(x, y, swt, t_indices)

    if standardize:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train, sample_weight=swt_train)

    trace = Ridge(alpha=alphas, fit_intercept=intercept, max_iter=100000)

    bout = []
    mse_out = []
    r2_train = []

    for alpha in alphas:
        trace.set_params(alpha=alpha)
        trace.fit(x_train, y_train, sample_weight=swt_train)
        bout.append(trace.coef_.tolist())
        mse_out.append(mean_squared_error(y_train, trace.predict(x_train), sample_weight=swt_train))
        r2_train.append(trace.score(x_train, y_train, swt_train))

    result["bout"] = bout
    result["mse_out"] = mse_out
    result["r2_train"] = r2_train


def process_cv(x, y, swt, nfolds, state, alphas, t_indices, h_indices, result):
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error, r2_score
    x_train_all, y_train_all, swt_train_all = get_train_group(x, y, swt, t_indices)

    alpha_out = []
    mse_out = []
    r2_out = []
    mean_mse = []
    mean_r2 = []

    for alpha in alphas:
        folds = KFold(n_splits=nfolds, shuffle=True, random_state=state)
        mse = []
        r2 = []
        for fold_n, (train_index, val_index) in enumerate(folds.split(x_train_all, y_train_all, swt_train_all)):
            x_train, x_val = x_train_all[train_index], x_train_all[val_index]
            y_train, y_val = y_train_all[train_index], y_train_all[val_index]
            swt_train = swt_val = None
            if swt is not None:
                swt_train, swt_val = swt_train_all[train_index], swt_train_all[val_index]
            scaler = StandardScaler()
            scaler.fit(x_train, sample_weight=swt_train)
            x_train = scaler.transform(x_train)
            x_val = scaler.transform(x_val)
            model = Ridge(alpha=alpha, fit_intercept=intercept, max_iter=100000)
            model.fit(x_train, y_train, sample_weight=swt_train)

            y_pred_val = model.predict(x_val).reshape(-1, )
            mse.append(mean_squared_error(y_val, y_pred_val, sample_weight=swt_val))
            r2.append(r2_score(y_val, y_pred_val, sample_weight=swt_val))
        alpha_out.append(alpha)
        mse_out.append(mse)
        r2_out.append(r2)
        mean_mse.append(np.mean(mse))
        mean_r2.append(np.mean(r2))

    result["alpha_out"] = alpha_out
    result["mse_out"] = mse_out
    result["r2_out"] = r2_out
    result["mean_mse"] = mean_mse
    result["mean_r2"] = mean_r2

    best_mean_r2 = max(mean_r2)
    best_index = mean_r2.index(best_mean_r2)
    best_alpha = alpha_out[best_index]

    result["best_alpha"] = best_alpha
    result["best_mean_r2"] = best_mean_r2

    process_fit(x, y, swt, best_alpha, t_indices, h_indices, result)


def to_valid(val):
    if val is None:
        return val
    if isinstance(val, (list, tuple)):
        return [None if np.isnan(v) else v for v in val]
    return None if np.isnan(val) else val


def get_train_group(x, y, swt, indices):
    if indices is not None:
        x_train = x[indices]
        y_train = y[indices]
        swt_train = None if swt is None else swt[indices]
    else:
        x_train = x
        y_train = y
        swt_train = swt

    return x_train, y_train, swt_train


def flatten(values):
    ret = []
    for val in values:
        if isinstance(val, (list, tuple)):
            ret.extend(val)
        else:
            ret.append(val)
    return ret


def save_data(pred, resid, pred_values, resid_values, count):
    new_fields = []
    new_records = RecordData()

    if pred is not None:
        new_fields.append(
            create_new_field(pred, "double", Measure.SCALE, Role.INPUT))
        new_records.add_columns(to_valid(pred_values))
        new_records.add_type(RecordData.CellType.DOUBLE)

    if resid is not None:
        new_fields.append(
            create_new_field(resid, "double", Measure.SCALE, Role.INPUT))
        new_records.add_columns(to_valid(resid_values))
        new_records.add_type(RecordData.CellType.DOUBLE)

    put_variables(new_fields, count, 0, new_records.get_binaries(), None)


def create_scatter_plot(title, x_label, x_data, y_label, y_data):
    scatter_chart = Chart(title)
    scatter_chart.set_type(Chart.Type.Scatterplot)
    scatter_chart.set_x_axis_label(x_label)
    scatter_chart.set_x_axis_data(x_data)
    scatter_chart.set_y_axis_label(y_label)
    scatter_chart.set_y_axis_data(y_data)

    return scatter_chart


def create_regression_table(alpha, x_names, y_name, intl, result):
    regression_table = Table(intl.loadstring("regression_coefficients"), "Ridge Regression Coefficients")
    regression_table.update_title(footnote_refs=[0])
    regression_table.set_default_cell_format(decimals=3)
    regression_table.set_max_data_column_width(140)

    regression_table_columns = []

    if standardize:
        columns_standardizing_cell = Table.Cell(intl.loadstring("standardizing_values"))
        columns_standardizing_cell.add_footnote_refs(1)

        columns_standardizing_mean_cell = Table.Cell(intl.loadstring("mean"))
        columns_standardizing_std_dev_cell = Table.Cell(intl.loadstring("std_dev"))

        columns_standardizing_cell.add_descendants([columns_standardizing_mean_cell.get_value(),
                                                    columns_standardizing_std_dev_cell.get_value()])

        regression_table_columns.append(columns_standardizing_cell.get_value())

        columns_standardizing_coefficients_cell = Table.Cell(intl.loadstring("standardized_coefficients"))

        regression_table_columns.append(columns_standardizing_coefficients_cell.get_value())

    columns_unstandardizing_coefficients_cell = Table.Cell(intl.loadstring("unstandardized_coefficients"))
    regression_table_columns.append(columns_unstandardizing_coefficients_cell.get_value())

    regression_table.add_column_dimensions(intl.loadstring("statistics"),
                                                 False,
                                                 regression_table_columns)

    rows_alpha_cell = Table.Cell(alpha)
    rows_alpha_cell.set_default_cell_format(decimals=3)

    if intercept:
        rows_intercept_cell = Table.Cell(intl.loadstring("intercept"))
        rows_intercept_cell.add_footnote_refs(2)

        rows_alpha_cell.add_descendants(rows_intercept_cell.get_value())

    for name in x_names:
        rows_predictor_cell = Table.Cell(name)
        rows_alpha_cell.add_descendants(rows_predictor_cell.get_value())

    regression_table_rows = [rows_alpha_cell.get_value()]
    regression_table.add_row_dimensions(intl.loadstring("alpha"),
                                              descendants=regression_table_rows)

    if intercept:
        intercept_row = []
        if standardize:
            intercept_row.append(None)
            intercept_row.append(None)
            intercept_row.append(result["int_out"])
            intercept_row.append(result["raw_int_out"])
        else:
            intercept_row.append(result["int_out"])

        regression_table.add_cells(intercept_row)

    for i in range(len(x_names)):
        predictor_row = []
        if standardize:
            predictor_row.append(result["mean_out"][i])
            predictor_row.append(result["std_out"][i])
            predictor_row.append(result["bout"][i])
            predictor_row.append(result["raw_bout"][i])
        else:
            predictor_row.append(result["bout"][i])

        regression_table.add_cells(predictor_row)

    regression_table.add_footnotes(intl.loadstring("dependent_variable").format(y_name))
    if standardize:
        regression_table.add_footnotes(intl.loadstring("footnotes_1"))
    if intercept:
        regression_table.add_footnotes(intl.loadstring("footnotes_2"))

    return regression_table


def create_fit_output(y, x_names, x_fnotes, y_name, alphas, intl, output_json, result):
    model_summary_table = Table(intl.loadstring("model_summary"), "Model Summary")
    model_summary_table.update_title(footnote_refs=[0, 1])
    model_summary_table.set_default_cell_format(decimals=3)
    model_summary_table.set_max_data_column_width(140)

    model_summary_table.add_row_dimensions(intl.loadstring("alpha"), descendants=alphas)

    columns_train_cell = Table.Cell(intl.loadstring("training_r2"))
    model_summary_table_columns = [columns_train_cell.get_value()]
    model_summary_table_cells = [result["r2_train"]]
    if holdout:
        columns_holdout_cell = Table.Cell(intl.loadstring("holdout_r2"))
        model_summary_table_columns.append(columns_holdout_cell.get_value())

        model_summary_table_cells.append(result["r2_holdout"])

    model_summary_table.add_column_dimensions(intl.loadstring("statistics"),
                                              False,
                                              model_summary_table_columns)
    model_summary_table.add_cells(model_summary_table_cells)

    model_footnotes = intl.loadstring("model").format(", ".join(x_fnotes))
    model_summary_table.add_footnotes([intl.loadstring("dependent_variable").format(y_name),
                                       model_footnotes])

    if len(model_footnotes) > 100:
        """ Make the columns wider than the default so footnotes don't wrap so often """
        model_summary_table.set_min_data_column_width(200)
        model_summary_table.set_max_data_column_width(240)

    output_json.add_table(model_summary_table)

    regression_table = create_regression_table(alphas[0], x_names, y_name, intl, result)
    output_json.add_table(regression_table)

    if get_value("plot_observed"):
        observed_chart = create_scatter_plot(intl.loadstring("dependent_by_predicted_value").format(y_name),
                                             intl.loadstring("predicted_value"),
                                             result["train_pred"],
                                             y_name,
                                             y)

        """ If both holdout and training data are present, add a footnote to differentiate them"""
        if holdout:
            observed_chart.set_footnote(intl.loadstring("training_data"))

        output_json.add_chart(observed_chart)

        if holdout:
            holdout_chart = create_scatter_plot(
                intl.loadstring("dependent_by_predicted_value").format(y_name),
                intl.loadstring("predicted_value"),
                result["holdout_pred"],
                y_name,
                y
            )

            """ If both holdout and training data are present, add a footnote to differentiate them"""
            holdout_chart.set_footnote(intl.loadstring("holdout_data"))

            output_json.add_chart(holdout_chart)

    if get_value("plot_residual"):
        residual_chart = create_scatter_plot(intl.loadstring("residual_by_predicted_value"),
                                             intl.loadstring("predicted_value"),
                                             result["train_pred"],
                                             intl.loadstring("residual"),
                                             result["train_resid"])

        """ If both holdout and training data are present, add a footnote to differentiate them"""
        if holdout:
            residual_chart.set_footnote(intl.loadstring("training_data"))

        output_json.add_chart(residual_chart)

        if holdout:
            holdout_chart = create_scatter_plot(intl.loadstring("residual_by_predicted_value"),
                                                intl.loadstring("predicted_value"),
                                                result["holdout_pred"],
                                                intl.loadstring("residual"),
                                                result["holdout_resid"])

            """ If both holdout and training data are present, add a footnote to differentiate them"""
            holdout_chart.set_footnote(intl.loadstring("holdout_data"))

            output_json.add_chart(holdout_chart)


def create_trace_output(x_names, x_fnotes, y_name, alphas, intl, output_json, result):
    if trace_table_max_rows > 0:
        """ Create the Trace Results table """
        trace_results_table = Table(intl.loadstring("trace_results"), "Trace Results")
        trace_results_table.update_title(footnote_refs=[0, 1])
        trace_results_table.add_footnotes([intl.loadstring("dependent_variable").format(y_name),
                                           intl.loadstring("model").format(", ".join(x_fnotes))])
        trace_results_table.set_default_cell_format(width=40, decimals=3)

        """ Set up the row dimension """
        trace_results_table.add_row_dimensions(intl.loadstring("alpha"),
                                               descendants=alphas)
        """ Set up the columns - R Square, MSE, and the predictors """
        trace_results_table_columns = []

        # Add the "R Square" column
        r2_category = Table.Cell(intl.loadstring("r2"))
        trace_results_table_columns.append(r2_category.get_value())

        # Add the "MSE" column
        mse_category = Table.Cell(intl.loadstring("mse"))
        trace_results_table_columns.append(mse_category.get_value())

        # Add a column for each name in x_names
        for name in x_names:
            cols_predictor_cell = Table.Cell(name)
            trace_results_table_columns.append(cols_predictor_cell.get_value())

        trace_results_table.add_column_dimensions(intl.loadstring("statistics"),
                                                  False,
                                                  trace_results_table_columns)

        r2_cells = to_valid(result["r2_train"])  # R2 column values
        mse_cells = to_valid(result["mse_out"])  # MSE column values

        """ Convert the column data into row data. """
        trace_results_table_cells = []
        for idx in range(min(len(r2_cells), trace_table_max_rows)):
            cur_row = [to_valid(r2_cells[idx]), to_valid(mse_cells[idx])]
            for coefficient in result["bout"][idx]:
                cur_row.append(to_valid(coefficient))
            trace_results_table_cells.append(cur_row)

        trace_results_table.set_cells(trace_results_table_cells)

        output_json.add_table(trace_results_table)

    regression_line_chart = GplChart(intl.loadstring("regression_line_chart_title"))

    graph_dataset = "graphdataset"

    metric = get_value("alpha_metric") if is_set("alpha_metric") else None
    if metric == "LG10":
        scale = "log(dim(1), base(10))"
    else:
        scale = "linear(dim(1))"

    hide_legend = ""
    if len(x_names) > 50:
        if cycle_patterns:
            hide_legend = "GUIDE: legend(aesthetic(aesthetic.shape.interior), null())"
        else:
            hide_legend = "GUIDE: legend(aesthetic(aesthetic.color.interior), null())"

    if standardize:
        sub_footnote = intl.loadstring("bottom_footnote_1")
    else:
        sub_footnote = intl.loadstring("bottom_footnote_2")

    if cycle_patterns:
        scale_aesthetic = "aesthetic.shape.interior"
        element_aesthetic = "shape"
        exterior_color = ""
    else:
        scale_aesthetic = "aesthetic.color.interior"
        element_aesthetic = "color"
        exterior_color = " color.exterior(color),"

    gpl = ["SOURCE: s=userSource(id(\"{0}\"))".format(graph_dataset),
           "DATA: x=col(source(s), name(\"x\"))",
           "DATA: y=col(source(s), name(\"y\"))",
           "DATA: color=col(source(s), name(\"color\"), unit.category())",
           "GUIDE: axis(dim(1), label(\"{0}\"))".format(intl.loadstring("alpha")),
           "GUIDE: axis(dim(2), label(\"{0}\"))".format(intl.loadstring("predictor_coefficients")),
           "{0}".format(hide_legend),
           "GUIDE: text.title(label(\"{0}\"))".format(intl.loadstring("regression_line_chart_title")),
           "GUIDE: text.footnote(label(\"{0}\"))".format(intl.loadstring("training_data")),
           "GUIDE: text.subfootnote(label(\"{0}\"))".format(sub_footnote),
           "SCALE: {0}".format(scale),
           "SCALE: linear(dim(2), include(0))",
           "SCALE: cat(aesthetic({0}))".format(scale_aesthetic),
           "ELEMENT: line(position(x*y), {0}.interior(color), size(size.\"1pt\"))".format(element_aesthetic),
           "ELEMENT: point(position(x*y), {0}.interior(color),{1} size(size.\"3pt\"))".format(element_aesthetic, exterior_color)]

    regression_line_chart.add_gpl_statement(gpl)

    lines = len(x_names)

    x_axis_data = np.repeat(alphas, lines).tolist()

    y_axis_data = flatten(result["bout"])

    color_data = x_names * len(alphas)

    regression_line_chart.add_variable_mapping("x", x_axis_data, graph_dataset)
    regression_line_chart.add_variable_mapping("y", y_axis_data, graph_dataset)
    regression_line_chart.add_variable_mapping("color", color_data, graph_dataset)
    output_json.add_chart(regression_line_chart)

    mse_line_chart = GplChart(intl.loadstring("mse_line_chart_title"))
    graph_dataset = "graphdataset"

    gpl = ["SOURCE: s=userSource(id(\"{0}\"))".format(graph_dataset),
           "DATA: x=col(source(s), name(\"x\"))",
           "DATA: y=col(source(s), name(\"y\"))",
           "GUIDE: axis(dim(1), label(\"{0}\"))".format(intl.loadstring("alpha")),
           "GUIDE: axis(dim(2), label(\"{0}\"))".format(intl.loadstring("mse")),
           "GUIDE: text.title(label(\"{0}\"))".format(intl.loadstring("mse_line_chart_title")),
           "SCALE: {0}".format(scale),
           "SCALE: linear(dim(2), include(0,1))",
           "ELEMENT: line(position(x*y))",
           "ELEMENT: point(position(x*y))"]

    mse_line_chart.add_gpl_statement(gpl)

    mse_line_chart.add_variable_mapping("x", alphas, graph_dataset)
    mse_line_chart.add_variable_mapping("y", result["mse_out"], graph_dataset)
    output_json.add_chart(mse_line_chart)

    r2_line_chart = GplChart(intl.loadstring("r2_line_chart_title"))
    graph_dataset = "graphdataset"

    gpl = ["SOURCE: s=userSource(id(\"{0}\"))".format(graph_dataset),
           "DATA: x=col(source(s), name(\"x\"))",
           "DATA: y=col(source(s), name(\"y\"))",
           "GUIDE: axis(dim(1), label(\"{0}\"))".format(intl.loadstring("alpha")),
           "GUIDE: axis(dim(2), label(\"{0}\"))".format(intl.loadstring("r2")),
           "GUIDE: text.title(label(\"{0}\"))".format(intl.loadstring("r2_line_chart_title")),
           "SCALE: {0}".format(scale),
           "SCALE: linear(dim(2), include(0,1))",
           "ELEMENT: line(position(x*y))",
           "ELEMENT: point(position(x*y))"]

    r2_line_chart.add_gpl_statement(gpl)

    r2_line_chart.add_variable_mapping("x", alphas, graph_dataset)
    r2_line_chart.add_variable_mapping("y", result["r2_train"], graph_dataset)
    output_json.add_chart(r2_line_chart)


def create_cv_output(y, x_names, x_fnotes, y_name, nfolds, alphas, intl, output_json, result):
    best_model_summary_table = Table(intl.loadstring("best_model_summary"), "Best Model Summary")
    best_model_summary_table.update_title(footnote_refs=[0, 1, 2])
    best_model_summary_table.set_default_cell_format(decimals=3)
    best_model_summary_table.set_max_data_column_width(140)

    best_model_summary_table.add_row_dimensions(intl.loadstring("alpha"),
                                                descendants=[result["best_alpha"]])

    best_model_summary_table_columns = [intl.loadstring("number_of_crossvalidation_folds"),
                                        intl.loadstring("training_r2"),
                                        intl.loadstring("average_test_subset_r2")]
    if holdout:
        best_model_summary_table_columns.append(intl.loadstring("holdout_r2"))
    best_model_summary_table.add_column_dimensions(intl.loadstring("statistics"),
                                                   False,
                                                   best_model_summary_table_columns)

    best_model_summary_table_cells = []
    nfolds_cell = Table.Cell(nfolds)
    nfolds_cell.set_default_cell_format(decimals=0)
    best_model_summary_table_cells.append(nfolds_cell.get_value())

    best_model_summary_table_cells.append(to_valid(result["r2_train"]))
    best_model_summary_table_cells.append(to_valid(result["best_mean_r2"]))
    if holdout:
        best_model_summary_table_cells.append(to_valid(result["r2_holdout"]))

    best_model_summary_table.add_cells(best_model_summary_table_cells)

    best_model_summary_table.add_footnotes([intl.loadstring("dependent_variable").format(y_name),
                                            intl.loadstring("model").format(", ".join(x_fnotes))])

    output_json.add_table(best_model_summary_table)

    regression_table = create_regression_table(result["best_alpha"], x_names, y_name, intl, result)
    output_json.add_table(regression_table)

    print_value = get_value("print").lower()
    is_compare = "compare" == print_value
    is_verbose = "verbose" == print_value

    if is_compare or is_verbose:
        model_comparisons_table = Table(intl.loadstring("model_comparisons"), "Model Comparisons")
        if is_compare:
            footnote_refs = list(range(3))
        else:
            footnote_refs = list(range(2))
        model_comparisons_table.update_title(footnote_refs=footnote_refs)
        model_comparisons_table.set_default_cell_format(decimals=3)
        model_comparisons_table.set_max_data_column_width(140)

        if is_compare:
            table_data = sorted(zip(result["mean_r2"], result["alpha_out"], result["mean_mse"]),
                                reverse=True)
        else:
            table_data = sorted(zip(result["mean_r2"],
                                    result["alpha_out"],
                                    result["r2_out"],
                                    result["mean_mse"],
                                    result["mse_out"]), reverse=True)

        table_data = list(zip(*table_data))
        model_comparisons_table.add_row_dimensions(intl.loadstring("alpha"),
                                                   descendants=table_data.pop(1))

        model_comparisons_table_columns = []
        if is_compare:
            model_comparisons_table_columns.append(intl.loadstring("average_test_subset_r2"))
            model_comparisons_table_columns.append(intl.loadstring("average_test_subset_mse"))
        else:
            model_comparisons_table_columns.append(intl.loadstring("average_test_set_r2"))
            for fold in range(nfolds):
                model_comparisons_table_columns.append(intl.loadstring("fold_r2").format(fold + 1))

            model_comparisons_table_columns.append(intl.loadstring("average_test_set_mse"))
            for fold in range(nfolds):
                model_comparisons_table_columns.append(intl.loadstring("fold_mse").format(fold + 1))

        model_comparisons_table.add_column_dimensions(intl.loadstring("statistics"),
                                                      False,
                                                      model_comparisons_table_columns)

        table_data_cells = list(zip(*table_data))
        if is_compare:
            for row_cells in table_data_cells:
                model_comparisons_table.add_cells(row_cells)
        else:
            for row_cells in table_data_cells:
                model_comparisons_table.add_cells(flatten(row_cells))

        model_comparisons_table.add_footnotes(intl.loadstring("dependent_variable").format(y_name))
        model_comparisons_table.add_footnotes(intl.loadstring("model").format(", ".join(x_fnotes)))
        if is_compare:
            model_comparisons_table.add_footnotes(intl.loadstring("nfolds_with_n").format(nfolds))

        output_json.add_table(model_comparisons_table)

    if get_value("plot_observed"):
        observed_chart = create_scatter_plot(intl.loadstring("dependent_by_predicted_value").format(y_name),
                                             intl.loadstring("predicted_value"),
                                             result["train_pred"],
                                             y_name,
                                             y)

        """ If both holdout and training data are present, add a footnote to differentiate them"""
        if holdout:
            observed_chart.set_footnote(intl.loadstring("training_data"))

        output_json.add_chart(observed_chart)

        if holdout:
            holdout_chart = create_scatter_plot(
                intl.loadstring("dependent_by_predicted_value").format(y_name),
                intl.loadstring("predicted_value"),
                result["holdout_pred"],
                y_name,
                y
            )

            """ If both holdout and training data are present, add a footnote to differentiate them"""
            holdout_chart.set_footnote(intl.loadstring("holdout_data"))

            output_json.add_chart(holdout_chart)

    if get_value("plot_residual"):
        residual_chart = create_scatter_plot(intl.loadstring("residual_by_predicted_value"),
                                             intl.loadstring("predicted_value"),
                                             result["train_pred"],
                                             intl.loadstring("residual"),
                                             result["train_resid"])

        """ If both holdout and training data are present, add a footnote to differentiate them"""
        if holdout:
            residual_chart.set_footnote(intl.loadstring("training_data"))

        output_json.add_chart(residual_chart)

        if holdout:
            holdout_chart = create_scatter_plot(intl.loadstring("residual_by_predicted_value"),
                                                intl.loadstring("predicted_value"),
                                                result["holdout_pred"],
                                                intl.loadstring("residual"),
                                                result["holdout_resid"])

            """ If both holdout and training data are present, add a footnote to differentiate them"""
            holdout_chart.set_footnote(intl.loadstring("holdout_data"))

            output_json.add_chart(holdout_chart)

    metric = get_value("alpha_metric") if is_set("alpha_metric") else None
    if metric == "LG10":
        scale = "log(dim(1), base(10))"
        x_scale = Chart.Scale.Log
    else:
        scale = "linear(dim(1))"
        x_scale = Chart.Scale.Linear

    if get_value("plot_mse"):
        average_mse_chart = GplChart(intl.loadstring("average_mse_line_chart_title"))
        graph_dataset = "graphdataset"
        gpl = ["SOURCE: s=userSource(id(\"{0}\"))".format(graph_dataset),
               "DATA: x=col(source(s), name(\"x\"))",
               "DATA: y=col(source(s), name(\"y\"))",
               "GUIDE: axis(dim(1), label(\"{0}\"))".format(intl.loadstring("alpha")),
               "GUIDE: axis(dim(2), label(\"{0}\"))".format(intl.loadstring("average_mse")),
               "GUIDE: text.title(label(\"{0}\"))".format(intl.loadstring("average_mse_line_chart_title")),
               "SCALE: {0}".format(scale),
               "SCALE: linear(dim(2), include(0,1))",
               "ELEMENT: line(position(x*y), missing.wings(), size(size.\"1pt\"))",
               "ELEMENT: point(position(x*y), size(size.\"3pt\"))"]

        average_mse_chart.add_gpl_statement(gpl)

        average_mse_chart.add_variable_mapping("x", alphas, graph_dataset)
        average_mse_chart.add_variable_mapping("y", result["mean_mse"], graph_dataset)
        output_json.add_chart(average_mse_chart)

    if get_value("plot_r2"):
        average_r2_chart = GplChart(intl.loadstring("average_r2_line_chart_title"))
        graph_dataset = "graphdataset"

        gpl = ["SOURCE: s=userSource(id(\"{0}\"))".format(graph_dataset),
               "DATA: x=col(source(s), name(\"x\"))",
               "DATA: y=col(source(s), name(\"y\"))",
               "GUIDE: axis(dim(1), label(\"{0}\"))".format(intl.loadstring("alpha")),
               "GUIDE: axis(dim(2), label(\"{0}\"))".format(intl.loadstring("average_r2")),
               "GUIDE: text.title(label(\"{0}\"))".format(intl.loadstring("average_r2_line_chart_title")),
               "SCALE: {0}".format(scale),
               "SCALE: linear(dim(2), include(0,1))",
               "ELEMENT: line(position(x*y), missing.wings(), size(size.\"1pt\"))",
               "ELEMENT: point(position(x*y), size(size.\"3pt\"))"]

        average_r2_chart.add_gpl_statement(gpl)

        average_r2_chart.add_variable_mapping("x", alphas, graph_dataset)
        average_r2_chart.add_variable_mapping("y", result["mean_r2"], graph_dataset)

        output_json.add_chart(average_r2_chart)
