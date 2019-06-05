# `debug_tools`
## File Descriptions
* `error_tracking_tools.py` supports different norms to measure error.
    * `TimeIterator` iterator which can be used as a timer
    * `ErrorTracker` class which keeps track of errors in form of list, and gives them labels (e.g. error @ resolution, or error @ time)
    * `ErrorIntegrator` class which keeps track of errors by summing them up.
* `visualization.py` contains the class `WindowManager`, which supports the following features
    * can display a state variable, given a state &rarr; `display_state`
    * can display an `error_tracker`, i.e. its list of errors displayed over its list of labels.
    This means it can list errors over time, or errors over resolution. &rarr; `display_error`
    * can draw lines which show how steep a certain line of a certain order would be &rarr; `draw_loglog_oder_line`