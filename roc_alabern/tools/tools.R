r.plot.showNumber <- function(df_input, row) {
  m = df_input[row,-1,with=FALSE]
  m = matrix(255-rev(as.numeric(m)), ncol=28)
  m = t(m[28:1, 1:28])
  par.default = par()$mar
  par(mar = c(0.14, 0, 1.3, 0))
  plot.new()
  title(main = paste0("row = ",row," | target = ",df_input[row,1,with=FALSE]))
  image(round(m), col = rplot::r.color.gradient.palette(c("black", "white"), levels = 100), add=TRUE)
  box()
  par(mar = par.default)
}

# r.plot.showNumber(df, 8)

