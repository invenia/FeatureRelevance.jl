# Some utility functions for differentiating different inputs
_validate(data) = data
_validate(data::AbstractVector) = reshape(data, (length(data), 1))
_get_names(data) = Tables.istable(data) ? Tables.columnnames(data) : 1:size(data, 2)
_get_columns(data) = Tables.istable(data) ? Tables.columns(data) : eachcol(data)
