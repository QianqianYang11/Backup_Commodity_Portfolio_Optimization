' ============================================================================================================================================================
' ============================================================================================================================================================
' This add-in estiamte the DCC-RGARCH or the DCC-GARCH model presented in Fiszeder, P., Faldzinski, M., Molnar, P., (2019), Range-based DCC models for covariance and value-at-risk forecasting, Journal of Empirical Finance, 54, 58-76, https://doi.org/10.1016/j.jempfin.2019.08.004.
' Additionaly, the conditional variances, the conditional covariances and the conditional correlations are saved as series. One-step-ahead forecast for the conditional variances, covariances and correlation are obatained. See documentation for more datails.
' Name of the add-in: DCC-RGARCH
' version: 1.00 (16.02.2021)
' Author: Marcin Faldzinski
' e-mail: marf@umk.pl
' ============================================================================================================================================================
' ============================================================================================================================================================

logmode 

' Default values for the parameters

'  boolean for  whether to activate Graphical User Interface, 1 - yes, 0 - no
!start_gui=1

' Models to estimate: 1 - denotes DCC-RGARCH, 2 - denotes DCC-GARCH,  3 - denotes DCC-GARCH-type(Built-in EViews GARCH-type model)
!radio_model=1

'Constraints on the ALPHA parameter (DCC model)
%constraints_a_dcc=""

'Constraints on the BETA parameter (DCC model)
%constraints_b_dcc=""

'Constraints on the degrees of freedom parameter (DCC model)
%constraints_v_dcc=""

' Initial values of the DCC parameters
%dcc_a_start="0.02"
%dcc_b_start="0.95"
%dcc_v_start="7"

' boolean for whether to perform the grid search over the set of starting values of the parameters for the DCC model (1 - perform grid sereach, 0 - do not)
!grid_search=0

' boolean for whether to the returns are multiply by 100 or not (1 - multiplied by 100, 0 - not)
!check_multiply=0

' boolean for whether to include constants in the mean equations (1 - include constants, 0 - do not)
!check_const_mean=1

'  boolean for whether to include constants in the variances equations (1 - include constants, 0 - do not)
!check_const_variance=1

'Optimization settings for DCC

'Convergence criteria
%conv_crit="0.00001"

' Hessian approximation method for DCC model
!hess_approx=1
%hess_appro="BFGS OPG Numeric"

' Hessian approximation method for (R)GARCH model
!hess_approx_g=1

' The step method
!step_m=1

' The maximum number of iterations
%max_iter="1000"

' The initial trust region size
%trust_value="0.25"

'  boolean for choosing the error distribution for (R)GARCH model (1 - Student's t-distribution, 2 - normal distribution)
!radio_tdist=2

'  boolean for choosing the error distribution for DCC model (1 - Student's t-distribution, 2 - normal distribution)
!radio_tdist_dcc=2

'  boolean for whether to obtain robust standard errors for DCC model or standard errors (1 - standard errors, 0 - robust standard errors)
!rob_err=1

' Whether to estimate built-in EViews GARCH-type model specifications
!ind_garch=1

' Built-in EViews GARCH-type model specification of each time series of returns
%ev_garch=""

' boolean for whether to show the output table with the estimation results or not (1 - show, 0 - do not)
!show_out=1

' boolean for whether to overwrite the existing output table with the estimation results or not (1 - overwrite, 0 - creates a new table with next available name)
!overwrite=1

' parameters values for the dcc_rgarch function (no GUI)

if @length(@equaloption("dep"))>0 then
	%dep=@equaloption("dep")
	%dep_high=@equaloption("dep_high")
	%dep_low=@equaloption("dep_low")

	if @wcount(@wfname)=0 then
		@uiprompt("Error: No active workfile exists", "O")
		stop
	endif

	%model=@equaloption("model")
	if	%model="rgarch" then
		!radio_model=1
	else
		if %model="garch" then
			!radio_model=2
		else
			if %model="ev_garch" then
				!radio_model=3
			else
			endif
		endif
	endif
	if @length(%model)=0 then
		!radio_model=1
	endif

	%ev_garch=@equaloption("ev_garch")

	if @length(@equaloption("tdist_garch"))>0 then	
		if @equaloption("tdist_garch")="yes" then		
			!radio_tdist=1
		endif
	endif

	if @length(@equaloption("tdist_dcc"))>0 then	
		if @equaloption("tdist_dcc")="yes" then
			!radio_tdist_dcc=1
		endif
	endif

	if @equaloption("std_err")="robust" then
		!rob_err=2
		else
		!rob_err=1
	endif
	
	if @equaloption("mean")="no" then
		!check_const_mean=0
	endif

	if @equaloption("variance")="no" then
		!check_const_variance=0
	endif

	if @length(@equaloption("smpl"))>0 then
		%edit_smpl=@equaloption("smpl")
	else
		%edit_smpl=@pagesmpl
	endif

	if @length(@equaloption("dcc_a"))>0 then
		%dcc_a_start=@equaloption("dcc_a")
	endif
	if @length(@equaloption("dcc_b"))>0 then
		%dcc_b_start=@equaloption("dcc_b")
	endif
	if @length(@equaloption("dcc_v"))>0 then
		%dcc_v_start=@equaloption("dcc_v")
	endif

	if @length(@equaloption("constraints_a_dcc"))>0 then
		%constraints_a_dcc=@equaloption("constraints_a_dcc")
	endif
	if @length(@equaloption("constraints_b_dcc"))>0 then
		%constraints_b_dcc=@equaloption("constraints_b_dcc")
	endif
	if @length(@equaloption("constraints_v_dcc"))>0 then	
		%constraints_v_dcc=@equaloption("constraints_v_dcc")
	endif

	if @length(@equaloption("grid_search"))>0 then
		if @equaloption("grid_search")="yes" then
			!grid_search=1
		endif
	endif

	if @length(@equaloption("convergence"))>0 then
	%conv_crit=@equaloption("convergence")
	endif

	if @equaloption("hess_approx_dcc")="BFGS" then
		!hess_approx=1
	else
		if @equaloption("hess_approx_dcc")="OPG" then
			!hess_approx=2'
		else
			if @equaloption("hess_approx_dcc")="Numeric" then
				!hess_approx=3
			endif
		endif
	endif

	if @equaloption("hess_approx_rgarch")="Hessian" then
		!hess_approx_g=1
	else
		if @equaloption("hess_approx_rgarch")="OPG" then
			!hess_approx_g=2'
		endif
	endif

	if @equaloption("step")="Marquardt" then
		!step_m=1
	else
		if @equaloption("step")="Dogleg" then
			!step_m=2'
		else
			if @equaloption("step")="Line_search" then
				!step_m=3
			endif
		endif
	endif

	if @length(@equaloption("max_ite"))>0 then
		%max_iter=@equaloption("max_ite")
	endif

	if @length(@equaloption("trust"))>0 then
		%trust_value=@val(@equaloption("trust"))
	endif

		if @equaloption("show_out")="yes" then
			!show_out=1
		else
			!show_out=0
		endif
		
		if @equaloption("overwrite")="yes" then
			!overwrite=1
		else
			!overwrite=0
		endif

	if @equaloption("multiply")="yes" then
		!check_multiply=1
	else
		!check_multiply=0
	endif

	!start_gui=0						' default, GUI is not shown if process dcc_rgarch is called
endif

if !start_gui=1 then

' check whether there is an active workfile
	if @wcount(@wfname)=0 then
		@uiprompt("Error: No active workfile exists", "O")
		stop
	endif

' if no sample specified get the current sample for the active page
	if @isempty(%edit_smpl) then
		%edit_smpl=@pagesmpl
	endif

' GUI and the parameters 
	!ret_1 = @uidialog("caption", "DCC(1,1)-(R)GARCH(1,1) estimation (1 of 2)", _
	"edit", %dep, "Enter the names of time series of dependent variables (returns)", 500, _
	"check", !check_multiply, "Returns are multiplied by 100", _
	"edit", %dep_high, "Enter names of high prices time series (need to be in the same order as for returns, required only for RGARCH model)", 500, _
	"edit", %dep_low, "Enter names of low prices time series (need to be in the same order as for returns, required only for RGARCH model)", 500, _
	"radio", !radio_model, "Choose model to estimate", "DCC-RGARCH DCC-GARCH DCC-GARCH-type(Built-in EViews GARCH-type model)", _
	"edit", %ev_garch, "Enter GARCH-type specifications for each time series - in parentheses and delimited with semicolon and space e.g. (arch(1,1) returns_1 c); (arch(1,2, tdist) returns_2 c) (required only for the DCC-GARCH-type model)", 2000, _
	"edit", %edit_smpl, "Sample", 50, _
	"colbreak", _
	"radio", !radio_tdist, "Choose error distribution for (R)GARCH model", "t-distribution Normal", _
	"radio", !radio_tdist_dcc, "Choose the multivariate conditional distribution of the error term in the DCC part", "t-distribution Normal", _
	"radio", !rob_err, "Choose type of errors", "Standard Robust_(Huber-White)", _
	"check", !check_const_mean, "Constants in the mean equations", _
	"check", !check_const_variance, "Constants in the variance equations", _
	"edit", %dcc_a_start, "Starting value of the ALPHA parameter (DCC model)", _
	"edit", %dcc_b_start, "Starting value of the BETA parameter (DCC model)", _
	"edit", %dcc_v_start, "Starting value of the degrees of freedom for t-distribution parameter (DCC model)", _
	"check", !grid_search, "Grid search over the set of starting values of the parameters (DCC model)", _
	"edit", %constraints_a_dcc, "Constraints on the ALPHA parameter (DCC model) e.g. 0 1", _
	"edit", %constraints_b_dcc, "Constraints on the BETA parameter (DCC model) e.g. 0 1", _
	"edit", %constraints_v_dcc, "Constraints on the degrees of freedom parameter for t-distribution (DCC model) e.g. 2 120")

	if !ret_1=-1 then
		stop
	endif

	!ret_2=@uidialog("caption", "DCC(1,1)-(R)GARCH(1,1) estimation (2 of 2)", _
	"edit", %conv_crit, "The convergence criterion", _
	"edit", %max_iter, "The maximum number of iterations", _
	"edit", %trust_value, "The initail trust region size", _
	 "radio", !hess_approx, "Choose Hessian approximation method for DCC model", "BFGS OPG Numeric", _
	"radio", !hess_approx_g, "Choose Hessian approximation method for (R)GARCH model", "Hessian OPG", _
	 "radio", !step_m, "Choose step method", "Marquardt Dogleg Line_search", _
	"check", !show_out, "Show output table", _
	"check", !overwrite, "Overwrite the existing output table (otherwise a new table will be created)")

	if !ret_2=-1 then
		stop
	endif

endif

' checking for errors in parameters

if @ispanel=1 then
	@uiprompt("Error: add-in cannot be used for panel workfiles", "O")
	stop
endif

if !radio_model=3 then
	if @isempty(%ev_garch) then
		@uiprompt("Error: No specifications for GARCH-type model entered", "O")
		stop
	endif
endif

if @isna(@val(%conv_crit)) then
	@uiprompt("Error: The convergence criteria not specified", "O")
	stop
endif

if @val(%max_iter)<1 then
	@uiprompt("Error: The maxium number of iterations cannot be negative", "O")
	stop
endif

if @isna(@val(%max_iter)) then
	@uiprompt("Error: The maxium number of iterations not specified ", "O")
	stop
endif

if @isna(@val(%trust_value)) then
	@uiprompt("Error: The initial trust region size not specified", "O")
	stop
endif

!tol=@val(%conv_crit)
!max_ite=@val(%max_iter)
!trust_val=@val(%trust_value)

if !hess_approx=1 then
	%hess_appro="BFGS"
else 
	if !hess_approx=2 then
		%hess_appro="OPG"
	else
		if !hess_approx=3 then
			%hess_appro="Numeric"
		endif
	endif
endif

if !hess_approx_g=1 then
	%hess_appro_g="Hessian"
else
	%hess_appro_g="OPG"
endif

if !step_m=1 then
	%step_method="Marquardt"
else
	if !step_m=2 then
		%step_method="Dogleg"
	else
		if !step_m=3 then
			%step_method="Linesearch"
		endif
	endif
endif

' type of variance-covariance method
if !rob_err=2 then
	%rob_cov="huber"
else
	%rob_cov="ordinary"
endif

if @wcount(%dep)=0 then
	@uiprompt("Error: No dependent variables specified", "O")
	stop
endif

if !radio_model=1 then
	if @wcount(%dep_high)<>@wcount(%dep) then
		@uiprompt("Error: There has to be exactly equal number of dependant variables (returns) and high prices." + @chr(13) + "The number of the time series with returns: " +@str(@wcount(%dep)) + @chr(13) + "The number of the time series with high prices: "+@str(@wcount(%dep_high)), "O")
		stop
	endif

	if @wcount(%dep_low)<>@wcount(%dep) then
		@uiprompt("Error: There has to be exactly equal number of dependant variables (returns) and low prices.  + @chr(13) +  "The number of the time series with returns: " +@str(@wcount(%dep)) + @chr(13) + "The number of the time series with low prices: "+@str(@wcount(%dep_low)), "O")
		stop
	endif
endif

' the matrix with constraints on the DCC model parameters
matrix(3,2) dcc_param_constraints=0

if @wcount(%constraints_a_dcc)=2 or @wcount(%constraints_b_dcc)=2 or @wcount(%constraints_a_dcc)=0 or @wcount(%constraints_b_dcc)=0 then
	if @wcount(%constraints_a_dcc)=2 and @wcount(%constraints_b_dcc)=2 then
		if @isna(@val(@word(%constraints_b_dcc,1))) or @isna(@val(@word(%constraints_b_dcc,2))) or @isna(@val(@word(%constraints_a_dcc,2))) or @isna(@val(@word(%constraints_a_dcc,2))) then
			@uiprompt("Error: The constraints on the ALPHA and/or BETA parameters are not numeric", "O")
			stop
		endif
		!cnst_dcc=1																											' scalar specifing  what kind of constraints are set; restrictions on both the ALPHA and the BETA parameter
		dcc_param_constraints(1,1)=@val(@word(%constraints_a_dcc, 1))
		dcc_param_constraints(1,2)=@val(@word(%constraints_a_dcc, 2))
		dcc_param_constraints(2,1)=@val(@word(%constraints_b_dcc, 1))
		dcc_param_constraints(2,2)=@val(@word(%constraints_b_dcc, 2))
	else
		if @wcount(%constraints_a_dcc)=2 and @wcount(%constraints_b_dcc)=0 then
			if @isna(@val(@word(%constraints_a_dcc,2))) or @isna(@val(@word(%constraints_a_dcc,2))) then
				@uiprompt("Error: The constraints on the ALPHA parameter are not numeric", "O")
				stop
			endif
			!cnst_dcc=2																										' restrictions only on the ALPHA parameter
			dcc_param_constraints(1,1)=@val(@word(%constraints_a_dcc,1))
			dcc_param_constraints(1,2)=@val(@word(%constraints_a_dcc,2))
		else
			if @wcount(%constraints_a_dcc)=0 and @wcount(%constraints_b_dcc)=2 then
				if @isna(@val(@word(%constraints_b_dcc,1))) or @isna(@val(@word(%constraints_b_dcc,2))) then
					@uiprompt("Error: The constraints on the BETA parameter are not numeric", "O")
				stop
				endif
				!cnst_dcc=3																									' restrictions only on the BETA parameter
				dcc_param_constraints(2,1)=@val(@word(%constraints_b_dcc,1))
				dcc_param_constraints(2,2)=@val(@word(%constraints_b_dcc,2))
			else
				!cnst_dcc=0																									' no restrictions on the ALPHA and the BETA parameters 
			endif
		endif
	endif
else
	@uiprompt("Error: Constraints on the parameters are not valid", "O")
	stop
endif

if @wcount(%constraints_v_dcc)=2 then
	if @isna(@val(@word(%constraints_v_dcc,1))) or @isna(@val(@word(%constraints_v_dcc,2))) then
		@uiprompt("Error: The constraints on the BETA parameters are not numeric", "O")
		stop
	endif
	!cnst_dcc_t=1																												' whether there are constraints on the degrees of freedom parameter, restrictions only on the degrees of freedom parameter
	dcc_param_constraints(3,1)=@val(@word(%constraints_v_dcc,1))
	dcc_param_constraints(3,2)=@val(@word(%constraints_v_dcc,2))
else
	if @wcount(%constraints_v_dcc)=0 then
		!cnst_dcc_t=0																											' no restrictions on the degrees of freedom parameter
	else
		@uiprompt("Error: Constraints on the degrees of freedom parameter are not valid", "O")
		stop
	endif
endif

if @val(@word(%constraints_a_dcc,1))>=@val(@word(%constraints_a_dcc,2)) or @val(@word(%constraints_b_dcc,1))>=@val(@word(%constraints_b_dcc,2)) or @val(@word(%constraints_v_dcc,1))>=@val(@word(%constraints_v_dcc,2)) then
	@uiprompt("Error: The upper bound has to be greater than the lower bound for the parameters constraints", "O")
	stop
endif

if !cnst_dcc=1 then
	if @val(%dcc_a_start)>dcc_param_constraints(1,2) or @val(%dcc_a_start)<dcc_param_constraints(1,1) then
		@uiprompt("Error: The starting value of the parameter ALPHA is outside the constraints", "O")
		stop
	endif
	if @val(%dcc_b_start)>dcc_param_constraints(2,2) or @val(%dcc_b_start)<dcc_param_constraints(2,1) then
		@uiprompt("Error: The starting value of the parameter BETA is outside the constraints", "O")
		stop
	endif
else
	if !cnst_dcc=2 then
		if @val(%dcc_a_start)>dcc_param_constraints(1,2) or @val(%dcc_a_start)<dcc_param_constraints(1,1) then
			@uiprompt("Error: The starting value of the parameter ALPHA is outside the constraints", "O")
			stop
		endif	
	else
		if !cnst_dcc=3 then
			if @val(%dcc_b_start)>dcc_param_constraints(2,2) or @val(%dcc_b_start)<dcc_param_constraints(2,1) then
				@uiprompt("Error: The starting value of the parameter BETA is outside the constraints", "O")
				stop
			endif
		endif
	endif
endif

if !cnst_dcc_t=1 then
	if @val(%dcc_v_start)>dcc_param_constraints(3,2) or @val(%dcc_v_start)<dcc_param_constraints(3,1) then
		@uiprompt("Error: The starting value of the degrees of freedom parameter is outside the constraints", "O")
		stop
	endif
endif

if @isempty(%edit_smpl) then
	%edit_smpl=@pagesmpl
else
	if @wcount(%edit_smpl)>2 then
		@uiprompt("Error: Sample has too many arguments", "O")
		stop
	endif

	if @wcount(%edit_smpl)<2 then
		@uiprompt("Error: Sample has too few arguments", "O")
		stop
	endif
endif

if !radio_model=3 then
	%ev_garch_ind=@wdrop(%ev_garch, ";")
	if @length(%ev_garch)=0 then
		@uiprompt("Error: No model specifications entered for built-in EViews GARCH-type models" + @chr(13) +"Choosing the DCC-GARCH-type model to estimate it is required to type in individual specifications", "O")
		stop
	endif

	if @wcount(%dep)<>@wcount(%ev_garch_ind) then
			@uiprompt("Error: There has to be exactly equal number of the dependent variables and model specifications for built-in EViews GARCH-type models."  + @chr(13) + "The number of the dependant variables: " +@str(!num_var)+  @chr(13) +  "The number of model specifications: " +@str(@wcount(%ev_garch_ind)), "O")
		stop
	endif
endif

' sample selection
%dcc_sms=@word(%edit_smpl, 1)
%dcc_sme=@word(%edit_smpl, 2)

smpl %dcc_sms %dcc_sme

series temp_s=@trend+1										' series that increases by one for each observation of the workfile, starting from 1
scalar s_first=@ifirst(temp_s)								' scalar with the first observation number
scalar s_last=@ilast(temp_s)								' scalar with the last observation number

' the number of obervations in the specified sample
scalar num_obs=@ilast(temp_s)-@ifirst(temp_s)+1
' The number of the time series with returns specified in %dep
!num_var=@wcount(%dep)

svector(27) dcc_settings										' string vector with variety of settings

dcc_settings(1)=@str(!radio_model)						' model to estimate 1 - RGARCH, 2 - GARCH, 3 - GARCH-type model built in EViews
dcc_settings(2)=%dcc_sms									' first observation in the sample
dcc_settings(3)=%dcc_sme									' last observation in the sample
dcc_settings(4)=@str(!radio_tdist)							' error distribution for the (R)GARCH model, 1 - Student's t-distribution, 2 - normal distribution
dcc_settings(5)=@str(!radio_tdist_dcc)					' error distribution for the DCC model, 1 - Student's t-distribution, 2 - normal distribution
dcc_settings(6)=@str(!rob_err)								' method for obtaining coefficients errors, 1 - standard errors, 2 -robust
dcc_settings(7)=@str(!grid_search)						' grid search, 1 - grid search active, 0 - no grid search
dcc_settings(8)=@str(!tol)										' convergence criterion
dcc_settings(9)=@str(!max_ite)								' the maximum number of iterations
dcc_settings(10)=@str(!trust_val)							' trust region size
dcc_settings(11)=%hess_appro							' Hessian approximation method for DCC model, name of the method:BFGS, OPG or Numeric
dcc_settings(12)=%hess_appro_g							' Hessian approximation method for (R)GARCH model, name of the method: Hessian or OPG
dcc_settings(13)=%step_method							' step method
dcc_settings(14)="0"											' sample adjustment, 1 - sample is adjusted, 0 - no sample adjustment
dcc_settings(15)=@str(num_obs)							' number of observations in the sample
'dcc_settings(16)													' adjusted smpl range for the GARCH-type model built-in EViews
dcc_settings(17)=@str(!cnst_dcc)							' constraints on the parameters of the DCC model, 0 - no constraints, 1 - constraints on both parameters, ALPHA and BETA, 2 - constraints only on ALPHA parameter, 3 - constraints only on BETA parameter,
dcc_settings(18)=@str(!cnst_dcc_t)						' constraints on the degrees of freedom of the DCC model, 0 - no constraints, 1 - constraints on the degrees of freedom parameter for Student's t-distribution
'dcc_settings(19)													' dimension of the matrix with the starting values for the grid search
dcc_settings(20)=@str(!hess_approx)					' Hessian approximation method for DCC model, 1 - BFGS, 2 - OPG, 3 - Numeric
dcc_settings(21)=@str(!hess_approx_g)					' Hessian approximation method for (R)GARCH model, 1 - Hessian, 2 - OPG
dcc_settings(22)=@str(!num_var)							' the number of time series with returns
dcc_settings(23)=@str(!check_const_mean)			' constants in the mean equations, 1 - include constants, 0 - do not include constants
dcc_settings(24)=@str(!check_const_variance)		' constants in the variance equations, 1 - include constants, 0 - do not include constants
'dcc_settings(25)													' the row number to insert tabel with results for the GARCH-type model built-in EViews
dcc_settings(26)=%rob_cov 									' type of variance-covariance method
dcc_settings(27)=@str(!check_multiply)					' whether returns are already multiplied by 100 or not

' temporary table TEMP_OUT for the estimation outputs
call del_obj("temp_out")
table temp_out

' starting time of the estimation process
!t_start=@now

' groups with the residuals and the conditional variances
call del_obj("res_g")
group res_g
call del_obj("ht_g")
group ht_g

' initial variance-covariance matrix
call del_obj("Ht_init")
matrix(!num_var, !num_var) Ht_init=0

' if there is the sample adjustment needed for DCC model then 1, otherwise 0
!smpl_adj=0

' the number of observations in the sample after adjustment, initially equals to the number of observations
!num_obs_adj=num_obs

' the row number to insert tabel with partial results into temporaray table, i.e. temp_out
!place_tab=0

'===================================================================================================================================
'===================================================================================================================================
'===================================================================================================================================
'===================================================================================================================================
'===================================================================================================================================
' MAIN EXECUTION OF THE PROGRAM
'===================================================================================================================================
'===================================================================================================================================
'===================================================================================================================================
'===================================================================================================================================
'===================================================================================================================================

' for every series in the %dep string (if a given series exists) run rgarch_est subrouting with the estimation of the RGARCH, GARCH and GARCH-type built-in EViews
!ite_m=0
for %ind_dep {%dep}
	!ite_m=!ite_m+1
	if @isobject(%ind_dep)=0 then
		@uiprompt("Error: " + %ind_dep + " does not exist in the active workfile")
		stop
	endif
	if !radio_model=1 then
		%high_name=@word(%dep_high,!ite_m)
		if @isobject(%high_name)=0 then
			@uiprompt("Error: " + %high_name + " does not exist in the active workfile")
			stop
		endif
		%low_name=@word(%dep_low,!ite_m)
		if @isobject(%low_name)=0 then
			@uiprompt("Error: " + %low_name + " does not exist in the active workfile")
			stop
		endif
	endif
	call rgarch_est(%ind_dep, !ite_m, dcc_settings)
next

' transform the group with residuals and with the conditional variances into matrices
stom(res_g, et_mat)
stom(ht_g, ht_mat)

call del_obj("dcc_smpl")

' adjusted sample range for the GARCH-type model built-in EViews
%dcc_smpl_adj=dcc_settings(16)				' adjusted smpl range

' check whether the number of observations in the matrix of residuals is the same as the starting number of observations
' if not then the sample is adjusted
if @rows(et_mat)<>num_obs then
	num_obs=@rows(et_mat)
	!smpl_adj=1
	%dcc_sms=@word(%dcc_smpl_adj,1)
	%dcc_sme=@word(%dcc_smpl_adj,2)
	sample dcc_smpl  %dcc_sms %dcc_sme
else
	!smpl_adj=0
	sample dcc_smpl  %dcc_sms %dcc_sme
endif

dcc_settings(2)=%dcc_sms						' first observation in the sample
dcc_settings(3)=%dcc_sme						' last observation in the sample
dcc_settings(14)=@str(!smpl_adj)				' sample adjustment - boolean
dcc_settings(15)=@str(num_obs)				' number of observations

if !radio_tdist_dcc=1 then
	vector(3) dcc_coefs_start
	vector(3) dcc_mlse
	vector(3) dcc_zstat
	vector(3) dcc_pval
	dcc_coefs_start(3)=@val(%dcc_v_start)
else
	vector(2) dcc_coefs_start
	vector(2) dcc_mlse
	vector(2) dcc_zstat
	vector(2) dcc_pval
endif

dcc_coefs_start(1)=@val(%dcc_a_start)
dcc_coefs_start(2) =@val(%dcc_b_start)

'call dcc_start(%dcc_a_start, %dcc_b_start, %dcc_v_start, !radio_tdist_dcc)

if !grid_search=1 then
	matrix(2,2) dcc_grid_start
	call dcc_grid(%dcc_a_start, %dcc_b_start, %dcc_v_start, !radio_tdist_dcc, !cnst_dcc, dcc_param_constraints, dcc_grid_start)
	dcc_settings(19)=@str(@rows(dcc_grid_start))				' dimension of the matrix with the starting values for the grid search
endif

' call the subroutine to obtain the outputs given so far (i.e estimation settings, the starting values, etc.)
call dcc_output1(!ite_m, dcc_param_constraints, dcc_coefs_start, dcc_settings)

' call the subroutine to obtain the parameters after the logit transformation (if neccessary) 
call dcc_param_trans(dcc_coefs_start, dcc_param_constraints, dcc_settings)

' call the subroutine to estimate the parameters of the DCC model (it takes into accounts the error distribution, grid search, robust errors of the coefficients)
call dcc_estim(dcc_coefs, et_mat, dcc_param_constraints, dcc_settings)

' call the subroutine to obtain errors of the coefficients based on the delta method and undoing transformation of the coefficients (if needed), addtionally vectors with z-statistics and p-values are obtained
call dcc_errors(dcc_coefs, dcc_mlse, dcc_zstat, dcc_pval, dcc_covar, dcc_param_constraints, dcc_settings)

' call the subroutine to obtain a one-period ahead forecasts of the conditional covariances and the conditional correlations
call dcc_forecasts(dcc_coefs, et_mat, ht_mat, dcc_settings)

' call the subroutine to obtain the final outputs
call dcc_output2(dcc_coefs, num_obs, dcc_logl, dcc_mlse, dcc_zstat, dcc_pval, !radio_tdist_dcc)

%endline="Completed successfully. Elapsed time: " +@str(!elapsed) + " seconds (or "+@str(!elapsed/60)+" minutes)"
statusline %endline

' cal the subroutine to place temporary output table (temp_out) into final output table (i.e. dcc_rgarch, dcc_garch or dcc_ev_garch)
call print_final_table(!radio_model, !place, !overwrite)

' clean up outputs
call del_obj("eq_init")
call del_obj("yi1yi2")
call del_obj("dcc_coefs_cnstr")
call del_obj("dcc_covar_cnstr")
call del_obj("dcc_grid_start")
call del_obj("dcc_coefs_un")
delete dcc_coefs_start dcc_d dcc_df dcc_h dcc_logl_vec dcc_q dcc_qf dcc_qs dcc_qsf dcc_r ht_mat num_obs temp_cov_* temp_rt_* res_g temp_s dcc_smpl ht_g s_first s_last temp_out dcc_param_constraints ev_garch_*_tab dcc_settings
'cor_unc cov_unc et_mat
'===================================================================================================================================
'===================================================================================================================================
' END EXECUTION OF THE PROGRAM 
'===================================================================================================================================
'===================================================================================================================================

'===================================================================================================================================
'===================================================================================================================================
' SUBROUTINES
'===================================================================================================================================
'===================================================================================================================================

' ============================================================================================================================
' subroutine for deleting objects (if a given object exists)

subroutine del_obj(string %obj_name)

if @isobject(%obj_name)=1 then
	delete %obj_name
endif
	
endsub

' ============================================================================================================================
' estimation of the RGARCH, GARCH or EV-GARCH models

subroutine rgarch_est(string %z, scalar !i1, svector dcc_settings)

!model=@val(dcc_settings(1))
!const_mean=@val(dcc_settings(23))
!const_variance=@val(dcc_settings(24))
!tdist=@val(dcc_settings(4))
!num_vars=@val(dcc_settings(22))
!max_ite=@val(dcc_settings(9))
!tol=@val(dcc_settings(8))
%hess_appro_g=dcc_settings(12)
%step_method=dcc_settings(13)
%rob_cov=dcc_settings(26)
!multi=@val(dcc_settings(27))

temp_out(2,1)="===================================================================================="
temp_out(3,1)="The number of the dependent variables: " + @str(!num_var) + " (i.e. " + @upper(%dep) +")"

if !model=1 then

	%s_line="Parameter estimation of the RGARCH model. Series name: " + @upper(%z)
	statusline %s_line
	call del_obj("parkinson_"+@str(!i1))

	' Parkinson's volatility estimator based on high and low prices
	%series_high=@word(%dep_high,!i1)
	%series_low=@word(%dep_low,!i1)
	if !multi=1 then
	series parkinson_{!i1}=((100*@log({%series_high}/{%series_low}))^2)/(4*@log(2))
	else
	series parkinson_{!i1}=(@log({%series_high}/{%series_low})^2)/(4*@log(2))
	endif
	temp_out(1,1)="DCC(1,1)-RGARCH(1,1) model, ADD-IN version: 1.00 by Marcin Faldzinski"

else
	if !model=2 then
		%s_line="Parameter estimation of the GARCH model. Series name: "+ @upper(%z)
		statusline %s_line
		temp_out(1,1)="DCC(1,1)-GARCH(1,1) model, ADD-IN version: 1.00 by Marcin Faldzinski"
	else
		%s_line="Parameter estimation of the GARCH-type model. Series name: " + @upper(%z)
		statusline %s_line
		temp_out(1,1)="DCC(1,1)-GARCH-type model built-in EViews, ADD-IN version: 1.00 by Marcin Faldzinski"
	endif
endif

if !model<3 then

' auxiliary series to indicate the first observation in-the-sample
call del_obj("d1")
series d1=0
d1(s_first)=1

' initial values of the parameters for RGARCH and GARCH models

	if !const_mean=1 then				' constant in the mean equation
		if !const_variance=1 then		' constant in the variance equation
			if !tdist=1 then						' !radio_tdist=1 means t-distributed errors
				equation eq_init.arch(tdist, m=500, c=1e-5) {%z} c
				coef(1) mu
				mu(1)=eq_init.c(1)	

				coef(1) ALPHA0
				ALPHA0(1)=eq_init.c(2)	

				coef(1) ALPHA1
				ALPHA1(1)=0.05

				coef(1) BETA
				BETA(1)=0.8

				coef(1) v
				v(1)=eq_init.c(5)

			else
				equation eq_init.arch(m=500, c=1e-5) {%z} c
				coef(1) mu
				mu(1)=eq_init.c(1)

				coef(1) ALPHA0
				if eq_init.c(2)>0 then
					ALPHA0(1)=eq_init.c(2)
				else
					ALPHA0(1)=0.02
				endif

				coef(1) ALPHA1
				ALPHA1(1)=0.05

				coef(1) BETA
				BETA(1)=0.8

			endif
		else
			if !tdist=1 then		' !radio_tdist=1 means t-distributed errors
				equation eq_init.arch(0,1, tdist, m=500, c=1e-5) {%z} c
					coef(1) mu
				mu(1)=eq_init.c(1)	

				coef(1) ALPHA1
				ALPHA1(1)=0.03

				coef(1) BETA
				BETA(1)=0.8

				coef(1) v
				v(1)=eq_init.c(4)
			else
				equation eq_init.arch(0,1, m=500, c=1e-5) {%z} c
				coef(1) mu
				mu(1)=eq_init.c(1)

				coef(1) ALPHA1
				ALPHA1(1)=0.05

				coef(1) BETA
				BETA(1)=0.8
			endif
		endif
	else
		if  !const_variance=1 then
			if !tdist=1 then		' !radio_tdist=1 means t-distributed errors
				equation eq_init.arch(tdist, m=500, c=1e-5) {%z}
				coef(1) ALPHA0
				ALPHA0(1)=eq_init.c(1)

				coef(1) ALPHA1
				if eq_init.c(2)>0 then
					ALPHA1(1)=eq_init.c(2)
				else
					ALPHA1(1)=0.01
				endif

				coef(1) BETA
				BETA(1)=0.8

				coef(1) v
				v(1)=eq_init.c(4)
			else
				equation eq_init.arch(m=500, c=1e-5) {%z}
				coef(1) ALPHA0
				ALPHA0(1)=eq_init.c(1)

				coef(1) ALPHA1
				ALPHA1(1)=0.05

				coef(1) BETA
				BETA(1)=0.8
			endif
		else
			if !tdist=1 then		' !radio_tdist=1 means t-distributed errors
				equation eq_init.arch(0,1, tdist, m=500, c=1e-5) {%z}

				coef(1) ALPHA1
				ALPHA1(1)=0.05

				coef(1) BETA
				BETA(1)=0.8

				coef(1) v
				v(1)=eq_init.c(3)
			else
				equation eq_init.arch(0,1, m=500, c=1e-5) {%z}

				coef(1) ALPHA1
				ALPHA1(1)=0.05

				coef(1) BETA
				BETA(1)=0.8
			endif
		endif
	endif

	call del_obj("yi1yi2")
	call del_obj("ymu_"+@str(!i1))
	call del_obj("loglike_"+@str(!i1))
	call del_obj("zsqt_"+@str(!i1))

	!i2=0
	for %z_left {%dep}
		!i2=!i2+1
		series yi1yi2=0
		yi1yi2={%z}*{%z_left}
		Ht_init(!i1,!i2)=@mean(yi1yi2)
	next

	if !model=1 then		'!radio_model=1 means RGARCH model
		call del_obj("rgarch_ht_"+@str(!i1)+@str(!i1))
		call del_obj("rgarch_res_"+@str(!i1))
		logl rgarch_{!i1}
		rgarch_{!i1}.append @logl loglike_{!i1}
			if !const_mean=1 then
				rgarch_{!i1}.append ymu_{!i1}={%z}-mu(1)
			else
				rgarch_{!i1}.append ymu_{!i1}={%z}
			endif
	else
		if !model=2 then			'!radio_model=2 means GARCH model
			call del_obj("garch_ht_"+@str(!i1))
			call del_obj("garch_res_"+@str(!i1))
			logl garch_{!i1}
			garch_{!i1}.append @logl loglike_{!i1}
			if !const_mean=1 then
				garch_{!i1}.append ymu_{!i1}={%z}-mu(1)
			else
				garch_{!i1}.append ymu_{!i1}={%z}
			endif
		else				' GARCH-type model built-in EViews
		endif
	endif

	if !model=1 then		'!radio_model=1 means RGARCH model
		if !const_variance=1 then		' !check_const_variance=1 means constant in the conditional variance equation
			rgarch_{!i1}.append rgarch_ht_{!i1}=@recode(d1=1, ht_init(!i1, !i1), ALPHA0(1)+ALPHA1(1)*parkinson_{!i1}(-1)+beta(1)*rgarch_ht_{!i1}(-1))
		else
			rgarch_{!i1}.append rgarch_ht_{!i1}=@recode(d1=1, ht_init(!i1, !i1), ALPHA1(1)*parkinson_{!i1}(-1)+beta(1)*rgarch_ht_{!i1}(-1))
		endif
	else
		if !model=2 then			'!radio_model=2 means GARCH model
			if !const_variance=1 then
				garch_{!i1}.append garch_ht_{!i1}=@recode(d1=1, ht_init(!i1, !i1), ALPHA0(1)+ALPHA1(1)*(ymu_{!i1}(-1))^2+beta(1)*garch_ht_{!i1}(-1))
			else
				garch_{!i1}.append garch_ht_{!i1}=@recode(d1=1, ht_init(!i1, !i1), ALPHA1(1)*(ymu_{!i1}(-1))^2+beta(1)*garch_ht_{!i1}(-1))
			endif
		else					' GARCH-type model built-in EViews
		endif
	endif


	if !tdist=1 then				' !radio_tdist=1 means t-distributed errors
		if !model=1 then		'!radio_model=1 means RGARCH model
			rgarch_{!i1}.append zsqt_{!i1}=(ymu_{!i1}/@sqrt(rgarch_ht_{!i1}))^2
			rgarch_{!i1}.append loglike_{!i1}=-0.5*@log(rgarch_ht_{!i1})  + @gammalog(0.5*(v(1)+1))-@gammalog(0.5*v(1))-0.5*1*@log(@acos(-1)*(v(1)-2))-0.5*(1+v(1))*@log(1+zsqt_{!i1}/(v(1)-2))
			rgarch_{!i1}.ml(showopts, m=!max_ite, c=!tol, covinfo=%hess_appro_g, optstep=%step_method, cov=%rob_cov)
			call del_obj("rgarch_"+@str(!i1)+"_tab")
			freeze(rgarch_{!i1}_tab) rgarch_{!i1}.output
		else
			if !model=2 then			'!radio_model=2 means GARCH model
				garch_{!i1}.append zsqt_{!i1}=(ymu_{!i1}/@sqrt(garch_ht_{!i1}))^2
				garch_{!i1}.append loglike_{!i1}=-0.5*@log(garch_ht_{!i1}) + @gammalog(0.5*(v(1)+1))-@gammalog(0.5*v(1))-0.5*1*@log(@acos(-1)*(v(1)-2))-0.5*(1+v(1))*@log(1+zsqt_{!i1}/(v(1)-2))
				garch_{!i1}.ml(showopts, m=!max_ite, c=!tol, covinfo=%hess_appro_g, optstep=%step_method, cov=%rob_cov)
				call del_obj("garch_"+@str(!i1)+"_tab")
				freeze(garch_{!i1}_tab) garch_{!i1}.output
			else					' GARCH-type model built-in EViews	
			endif
		endif
	else

		if !model=1 then		'!radio_model=1 means RGARCH model
			rgarch_{!i1}.append loglike_{!i1}=-0.5*(@log(rgarch_ht_{!i1})+ @log(2*@acos(-1)) + (ymu_{!i1}^2)/rgarch_ht_{!i1})
			rgarch_{!i1}.ml(showopts, m=!max_ite, c=!tol, covinfo=%hess_appro_g, optstep=%step_method, cov=%rob_cov)
			call del_obj("rgarch_"+@str(!i1)+"_tab")
			freeze(rgarch_{!i1}_tab) rgarch_{!i1}.output
		else
			if !model=2 then			'!radio_model=2 means GARCH model
				garch_{!i1}.append loglike_{!i1}=-0.5*(@log(garch_ht_{!i1})+ @log(2*@acos(-1)) + (ymu_{!i1}^2)/garch_ht_{!i1})
				garch_{!i1}.ml(showopts, m=!max_ite, c=!tol, covinfo=%hess_appro_g, optstep=%step_method, cov=%rob_cov)
				call del_obj("garch_"+@str(!i1)+"_tab")
				freeze(garch_{!i1}_tab) garch_{!i1}.output
			else					' GARCH-type model built-in EViews	
			endif
		endif
	endif
endif


if !model=3 then				' GARCH-type model built-in EViews
	svector garch_spec=@wsplit(@wdrop(%ev_garch, ";"))
	%garch_spec_{!i1}=@stripparens(garch_spec(!i1))
	equation ev_garch_{!i1}.{%garch_spec_{!i1}}
			
	ev_garch_{!i1}.makegarch ev_garch_ht_{!i1}
	ev_garch_{!i1}.makeresids(s) ev_garch_res_{!i1}

	res_g.add ev_garch_res_{!i1}
	ht_g.add ev_garch_ht_{!i1}

	if @obssmpl<@obsrange then
		smpl %dcc_sme+1 %dcc_sme+1
		ev_garch_{!i1}.forecast ev_garch_fcast_mu_{!i1} ev_garch_fcast_se_{!i1} ev_garch_fcast_var_{!i1}					
		matrix(1, !num_var) ev_garch_fcast_ht
		ev_garch_fcast_ht(1, !i1)=ev_garch_fcast_var_{!i1}(s_last+1)
	else
		pagestruct(end=@last+1) *
		smpl %dcc_sme+1 %dcc_sme+1
		ev_garch_{!i1}.forecast ev_garch_fcast_mu_{!i1} ev_garch_fcast_se_{!i1} ev_garch_fcast_var_{!i1}
		matrix(1, !num_var) ev_garch_fcast_ht
		ev_garch_fcast_ht(1, !i1)=ev_garch_fcast_var_{!i1}(s_last+1)	
		smpl %dcc_sms %dcc_sme
		pagestruct(end=@last-1) *
		smpl %dcc_sms %dcc_sme
	endif					

	call del_obj("ev_garch_"+@str(!i1)+"_tab")
	freeze(ev_garch_{!i1}_tab) ev_garch_{!i1}.output
	%garch_smpl_{!i1}=@right(ev_garch_{!i1}_tab(4,1), @length(ev_garch_{!i1}_tab(4,1))-@instr(ev_garch_{!i1}_tab(4,1),":")-1)

	if ev_garch_{!i1}.@regobs<!num_obs_adj then
		!num_obs_adj=ev_garch_{!i1}.@regobs
		dcc_settings(16)=%garch_smpl_{!i1}				' adjusted sample range
		dcc_settings(25)=@str(!place_tab)				'  the row number to insert tabel with results
	endif
		delete ev_garch_fcast_mu_* ev_garch_fcast_se_* ev_garch_fcast_var_*
		smpl %dcc_sms %dcc_sme
endif

' standardised residuals (residuals for GARCH-type model built-in EViews have already been obtained)
if !model=1 then
	series rgarch_res_{!i1}=(ymu_{!i1})/(rgarch_ht_{!i1}^0.5)
else
	if !model=2 then
		series garch_res_{!i1}=(ymu_{!i1})/(garch_ht_{!i1}^0.5)
	endif
endif

' one-period ahead conditional variance forecast
if !model=1 then
	res_g.add rgarch_res_{!i1}
	ht_g.add rgarch_ht_{!i1}
	if !const_variance=1 then
		matrix(1, !num_var) rgarch_fcast_ht
		rgarch_fcast_ht(1,!i1)=ALPHA0(1)+ALPHA1(1)*parkinson_{!i1}(s_last)+beta(1)*rgarch_ht_{!i1}(s_last)
	else
		matrix(1, !num_var) rgarch_fcast_ht
		rgarch_fcast_ht(1,!i1)=ALPHA1(1)*parkinson_{!i1}(s_last)+beta(1)*rgarch_ht_{!i1}(s_last)
	endif
else
	if !model=2 then
		res_g.add garch_res_{!i1}
		ht_g.add garch_ht_{!i1}
		if !const_variance=1 then
			matrix(1, !num_var) garch_fcast_ht
			garch_fcast_ht(1,!i1)=ALPHA0(1)+ALPHA1(1)*(ymu_{!i1}(s_last))^2+beta(1)*garch_ht_{!i1}(s_last)
		else
			matrix(1, !num_var) garch_fcast_ht
			garch_fcast_ht(1,!i1)=ALPHA1(1)*(ymu_{!i1}(s_last))^2+beta(1)*garch_ht_{!i1}(s_last)		
		endif
	else					' GARCH-type model built-in EViews forecasts have already been obtained
	endif
endif

' estimation output place into a temporary table

if !model=1 then
	temp_out(!i1*28-21,1)=""
		if !tdist=1 then
			temp_out(!i1*29-23,1)="========================================================================================="
			temp_out(!i1*29-22,1)="RGARCH(1,1) (t-distribution) for dependent variable: "+@upper(%z) + " (assigned number to the variable: "+@str(!i1) +")"
			tabplace(temp_out, rgarch_{!i1}_tab,  "A"+@str(!i1*29-21), "A1", "E32")
		else
			temp_out(!i1*28-23,1)="========================================================================================="
			temp_out(!i1*28-22,1)="RGARCH(1,1) (Normal distribution) for dependent variable: "+@upper(%z) + " (assigned number to the variable: "+@str(!i1) +")"
			tabplace(temp_out, rgarch_{!i1}_tab,  "A"+@str(!i1*28-21), "A1", "E32")
		endif
	delete rgarch_{!i1}_tab
else
	if !model=2 then
		temp_out(!i1*28-21,1)=""
			if !tdist=1 then
				temp_out(!i1*29-23,1)="========================================================================================="
				temp_out(!i1*29-22,1)="GARCH(1,1) (t-distribution) for dependent variable: "+@upper(%z) + " (assigned number to the variable: "+@str(!i1) +")"
				tabplace(temp_out, garch_{!i1}_tab,  "A"+@str(!i1*29-21), "A1", "E32")
			else
				temp_out(!i1*28-23,1)="========================================================================================="
				temp_out(!i1*28-22,1)="GARCH(1,1) (Normal distribution) for dependent variable: "+@upper(%z) + " (assigned number to the variable: "+@str(!i1) +")"
				tabplace(temp_out, garch_{!i1}_tab,  "A"+@str(!i1*28-21), "A1", "E32")
			endif
		delete garch_{!i1}_tab
	else 					' GARCH-type model built-in EViews
		!rows_garch_{!i1}=ev_garch_{!i1}_tab.@rows
			if !i1=1 then
				!place_tab=!place_tab+7
			else
				!i1b=!i1-1
				!place_tab=!place_tab+ev_garch_{!i1b}_tab.@rows+5
			endif
		temp_out(!place_tab-2,1)="========================================================================================="
		temp_out(!place_tab-1,1)="GARCH-type model built-in EViews for dependent variable: "+@upper(%z) + " (assigned number to the variable: "+@str(!i1) +")"
		tabplace(temp_out, ev_garch_{!i1}_tab,  "A"+@str(!place_tab), "A1", "E100")
			if !i1=!num_vars then
				!place_tab=!place_tab+ev_garch_{!i1}_tab.@rows
				dcc_settings(25)=@str(!place_tab)				'  the row number to insert tabel with results
			endif
	endif
endif

endsub

' ============================================================================================================================
' matrix with the grid of the starting parameters

subroutine local dcc_grid(string %dcc_a_start, string %dcc_b_start, string %dcc_v_start, scalar tdist_dcc, scalar cnst_dcc, matrix dcc_constraints, matrix dcc_grid_start)

!lb_a=dcc_constraints(1,1)
!ub_a=dcc_constraints(1,2)
!lb_b=dcc_constraints(2,1)
!ub_b=dcc_constraints(2,2)
!lb_v=dcc_constraints(3,1)
!ub_v=dcc_constraints(3,2)

if cnst_dcc=0 then
	!lb_a=0
	!ub_a=1
	!lb_b=0
	!ub_b=1
else
	if cnst_dcc=2 then
		!lb_b=0
		!ub_b=1
	else
		if cnst_dcc=3 then
			!lb_a=0
			!ub_a=1
		endif
	endif
endif

if tdist_dcc=1 then
	!lb_v=2
	!ub_v=100
endif

!width_a=!ub_a-!lb_a
!width_b=!ub_b-!lb_b
!width_v=!ub_v-!lb_v

!slices_b=40
!slices_a=10
!s_width_b=!width_b/!slices_b
!s_width_a=!width_a/!slices_a

!slices_v=10
!s_width_v=!width_v/!slices_v

if tdist_dcc=1 then
	!dim=(!slices_b-1)*(!slices_a-1)*(!slices_v-1)+1
	matrix(!dim, 3) dcc_grid
	dcc_grid(1,1)=@val(%dcc_a_start)
	dcc_grid(1,2)=@val(%dcc_b_start)
	dcc_grid(1,3)=@val(%dcc_v_start)
else
	!dim=(!slices_b-1)*(!slices_a-1)+1
	matrix(!dim, 2) dcc_grid
	dcc_grid(1,1)=@val(%dcc_a_start)
	dcc_grid(1,2)=@val(%dcc_b_start)
endif

!num_s=1

if tdist_dcc=1 then
	for !i=1 to !slices_b-1
		!beta_i=!ub_b-!s_width_b*!i
		!reminder_b=1-!beta_i
		if !reminder_b>!width_a then
			!slice_width_rem=!s_width_a/!slices_a
		else
			!slice_width_rem=!reminder_b/!slices_a
		endif

		for !j=1 to !slices_a-1
			for !k=1 to !slices_v-1
				!alpha_j=!lb_a+!slice_width_rem*!j
				if !alpha_j+!beta_i>=1 then exitloop
				endif
				!num_s=!num_s+1
				dcc_grid(!num_s, 2)=!beta_i
				dcc_grid(!num_s, 1)=!alpha_j
				dcc_grid(!num_s, 3)=!lb_v+!s_width_v/4*!k
			next
		next
	next
else

	for !i=1 to !slices_b-1
		!beta_i=!ub_b-!s_width_b*!i
		!reminder_b=1-!beta_i
		if !reminder_b>!width_a then
			!slice_width_rem=!s_width_a/!slices_a
		else
			!slice_width_rem=!reminder_b/!slices_a
		endif

		for !j=1 to !slices_a-1
			!alpha_j=!lb_a+!slice_width_rem*!j
			if !alpha_j+!beta_i>=1 then exitloop
			endif
			!num_s=!num_s+1
			dcc_grid(!num_s, 2)=!beta_i
			dcc_grid(!num_s, 1)=!alpha_j
		next
	next
endif

if !num_s<@rows(dcc_grid) then
	dcc_grid_start=@subextract(dcc_grid, 1, 1, !num_s)
else
	dcc_grid_start=dcc_grid
endif

endsub


' ============================================================================================================================
' DCC output table (before estimation)

subroutine dcc_output1(scalar pos, matrix dcc_constraints, vector dcc_param, svector dcc_settings)

!model=@val(dcc_settings(1))
!tdist=@val(dcc_settings(4))
!tdist_dcc=@val(dcc_settings(5))
!cnst_dcc=@val(dcc_settings(17))
!cnst_dcc_t=@val(dcc_settings(18))
!hess_approx=@val(dcc_settings(20))
!rob_err=@val(dcc_settings(6))
!grid_search=@val(dcc_settings(7))
!smpl_adj=@val(dcc_settings(14))
!place_ev_tab=@val(dcc_settings(25))	

if !model<3 then
	if !tdist=1 then
		!place=(pos+1)*29
	else
		!place=(pos+1)*28
	endif
else
	if !tdist=1 then
		!place=(pos+1)*30
	else
		!place=(pos+1)*29
	endif
endif

temp_out.setwidth(1) 26

if !model<3 then
	if !tdist_dcc=2 then
		temp_out(!place-22,1)="DCC(1,1) (Normal distribution)"
	else
		temp_out(!place-22,1)="DCC(1,1) (t-distribution)"
	endif
	temp_out.setjust(!place-22,1) left
	temp_out(!place-23,1)="========================================================================================="
	temp_out(!place-21,1)="Method: Maximum Likelihood (" + %hess_appro + " / " + %step_method + " steps)"
	temp_out(!place-20,1)="Date: " + @strnow("DD.MM.yyyy") +" Time: " + @strnow("HH:MI")
	temp_out.setjust(!place-19,1) left
	temp_out(!place-19,1)="Sample: " +%dcc_sms+" "+%dcc_sme
	temp_out(!place-18,1)="Included observations: "+@str(num_obs)
	temp_out(!place-17,1)="Evaluation order: By observation"
	temp_out(!place-16,1)="Estimation settings: tol= "+@str(!tol) + " derivs=numeric"
else
	if !model=3 then
		!place=!place_ev_tab+3+23
	endif

	if !tdist_dcc=2 then
		temp_out(!place-22,1)="DCC(1,1) (Normal distribution)"
	else
		temp_out(!place-22,1)="DCC(1,1) (t-distribution)"
	endif
	temp_out.setjust(!place-22,1) left
	temp_out(!place-23,1)="========================================================================================="
	temp_out(!place-21,1)="Method: Maximum Likelihood (" + %hess_appro + " / " + %step_method + " steps)"
	temp_out(!place-20,1)="Date: " + @strnow("DD.MM.yyyy") +" Time: " + @strnow("HH:MI")
	temp_out.setjust(!place-19,1) left

	if !smpl_adj=0 then
		temp_out(!place-19,1)="Sample: " +%dcc_sms+" "+%dcc_sme
		temp_out(!place-18,1)="Included observations: "+@str(num_obs)
	else
		temp_out(!place-19,1)="Sample (adjusted): " +%dcc_smpl_adj
		temp_out(!place-18,1)="Included observations: " +@str(num_obs) +" after adjustments"
	endif
	temp_out(!place-17,1)="Evaluation order: By observation"
	temp_out(!place-16,1)="Estimation settings: tol= "+@str(!tol) + " derivs=numeric"
endif

if !cnst_dcc=0 then
	if !tdist_dcc=1 then
		if !cnst_dcc_t=1 then
			temp_out(!place-15,1)="Initial Values: ALPHA=" +@str(dcc_param(1)) +", BETA=" +@str(dcc_param(2)) + ", T-DIST. DOF=" +@str(dcc_param(3)) +" with restrictions: " +@str(dcc_constraints(2,1))+"<=T-DIST. DOF<="+@str(dcc_constraints(2,2)) 
		else
			temp_out(!place-15,1)="Initial Values: ALPHA=" +@str(dcc_param(1)) +", BETA=" +@str(dcc_param(2)) + ", T-DIST. DOF=" +@str(dcc_param(3))
		endif
	else
		temp_out(!place-15,1)="Initial Values: ALPHA=" +@str(dcc_param(1)) +", BETA=" +@str(dcc_param(2))
	endif
		if !hess_approx=1 then
			if !rob_err=1 then
				if !cnst_dcc_t=1 then
					temp_out(!place-13,1)="Coefficient covariance computed using observed Hessian and the delta method"
				else
					temp_out(!place-13,1)="Coefficient covariance computed using observed Hessian"
				endif
			else
				if !cnst_dcc_t=1 then
					temp_out(!place-13,1)="Coefficient covariance computed using sandwich method (robust Huber-White) and the delta method"
				else
					temp_out(!place-13,1)="Coefficient covariance computed using sandwich method (robust Huber-White)"
				endif
			endif
	else
		if !hess_approx=2 then
			if !rob_err=1 then
				if !cnst_dcc_t=1 then
					temp_out(!place-13,1)="Coefficient covariance computed using outer product of gradients and the delta method"
				else
					temp_out(!place-13,1)="Coefficient covariance computed using outer product of gradients"
				endif
			else
				if !cnst_dcc_t=1 then				
					temp_out(!place-13,1)="Coefficient covariance computed using sandwich method (robust Huber-White) and the delta method"
				else
					temp_out(!place-13,1)="Coefficient covariance computed using sandwich method (robust Huber-White)"
				endif
			endif
		else
			if !rob_err=1 then
				if !cnst_dcc_t=1 then
					temp_out(!place-13,1)="Coefficient covariance computed using numerical approximation and the delta method"
				else
					temp_out(!place-13,1)="Coefficient covariance computed using numerical approximation"
				endif
			else
				if !cnst_dcc_t=1 then
					temp_out(!place-13,1)="Coefficient covariance computed using sandwich method (robust Huber-White) and the delta method"
				else
					temp_out(!place-13,1)="Coefficient covariance computed using sandwich method (robust Huber-White)"
				endif
			endif
		endif
	endif 
else
	if !cnst_dcc=1 then
		if !radio_tdist_dcc=1 then
			if !cnst_dcc_t=1 then
				temp_out(!place-15,1)="Initial Values: ALPHA=" +@str(dcc_param(1)) +", BETA=" +@str(dcc_param(2)) + ", T-DIST. DOF=" +@str(dcc_param(3)) +" with restrictions: " +@str(dcc_constraints(1,1))+"<=ALPHA<="+@str(dcc_constraints(1,2)) +", " +@str(dcc_constraints(2,1))+"<=BETA<=" +@str(dcc_constraints(2,2))+", " +@str(dcc_constraints(3,1))+"<=T-DIST. DOF<="+@str(dcc_constraints(3,2)) 
			else
				temp_out(!place-15,1)="Initial Values: ALPHA=" +@str(dcc_param(1)) +", BETA=" +@str(dcc_param(2)) + ", T-DIST. DOF=" +@str(dcc_param(3)) +" with restrictions: " +@str(dcc_constraints(1,1))+"<=ALPHA<="+@str(dcc_constraints(1,2)) +", " +@str(dcc_constraints(2,1))+"<=BETA<=" +@str(dcc_constraints(2,2))
			endif
		else
			temp_out(!place-15,1)="Initial Values: ALPHA=" +@str(dcc_param(1)) +", BETA=" +@str(dcc_param(2)) +" with restrictions: " +@str(dcc_constraints(1,1))+"<=ALPHA<="+@str(dcc_constraints(1,2)) +", " +@str(dcc_constraints(2,1))+"<=BETA<=" +@str(dcc_constraints(2,2))
		endif
		if !hess_approx=1 then
			if !rob_err=1 then
				temp_out(!place-13,1)="Coefficient covariance computed using observed Hessian and the delta method"
			else
				temp_out(!place-13,1)="Coefficient covariance computed using sandwich method (robust Huber-White) and the delta method"
			endif
		else
			if !hess_approx=2 then
				if !rob_err=1 then
					temp_out(!place-13,1)="Coefficient covariance computed using outer product of gradients and the delta method"
				else
					temp_out(!place-13,1)="Coefficient covariance computed using sandwich method (robust Huber-White) and the delta method"
				endif
			else
				if !rob_err=1 then
					temp_out(!place-13,1)="Coefficient covariance computed using numerical approximation and the delta method"
				else
					temp_out(!place-13,1)="Coefficient covariance computed using sandwich method (robust Huber-White) and the delta method"
				endif
			endif
		endif 
	else
		if !cnst_dcc=2 then
			if !radio_tdist_dcc=1 then
				if !cnst_dcc_t=1 then
					temp_out(!place-15,1)="Initial Values: ALPHA=" +@str(dcc_param(1)) +", BETA=" +@str(dcc_param(2))+ ", T-DIST. DOF=" +@str(dcc_param(3)) +" with restrictions: " +@str(dcc_constraints(1,1))+"<ALPHA<"+@str(dcc_constraints(1,2)) +", " +@str(dcc_constraints(3,1))+"<=T-DIST. DOF<="+@str(dcc_constraints(3,2)) 
				else
					temp_out(!place-15,1)="Initial Values: ALPHA=" +@str(dcc_param(1)) +", BETA=" +@str(dcc_param(2))+ ", T-DIST. DOF=" +@str(dcc_param(3)) +" with restrictions: " +@str(dcc_constraints(1,1))+"<=ALPHA<="+@str(dcc_constraints(1,2))
				endif
			else
				temp_out(!place-15,1)="Initial Values: ALPHA=" +@str(dcc_param(1)) +", BETA=" +@str(dcc_param(2)) +" with restrictions: " +@str(dcc_constraints(1,1))+"<=ALPHA<="+@str(dcc_constraints(1,2))
			endif
			if !hess_approx=1 then
				if !rob_err=1 then
					temp_out(!place-13,1)="Coefficient covariance computed using observed Hessian and the delta method"
				else
					temp_out(!place-13,1)="Coefficient covariance computed using sandwich method (robust Huber-White) and the delta method"
				endif
			else
				if !hess_approx=2 then
					if !rob_err=1 then
						temp_out(!place-13,1)="Coefficient covariance computed using outer product of gradients and the delta method"
					else
						temp_out(!place-13,1)="Coefficient covariance computed using sandwich method (robust Huber-White) and the delta method"
					endif
				else
					if !rob_err=1 then
						temp_out(!place-13,1)="Coefficient covariance computed using numerical approximation and the delta method"
					else
						temp_out(!place-13,1)="Coefficient covariance computed using sandwich method (robust Huber-White) and the delta method"
					endif
				endif
			endif 
		else
		if !radio_tdist_dcc=1 then
			if !cnst_dcc_t=1 then
				temp_out(!place-15,1)="Initial Values: ALPHA=" +@str(dcc_param(1)) +", BETA=" +@str(dcc_param(2)) + ", T-DIST. DOF=" +@str(dcc_param(3)) +" with restrictions: " +@str(dcc_constraints(2,1))+"<=BETA<=" +@str(dcc_constraints(2,2))+", " +@str(dcc_constraints(3,1))+"<=T-DIST. DOF<="+@str(dcc_constraints(3,2)) 
			else
				temp_out(!place-15,1)="Initial Values: ALPHA=" +@str(dcc_param(1)) +", BETA=" +@str(dcc_param(2)) + ", T-DIST. DOF=" +@str(dcc_param(3)) +" with restrictions: " +@str(dcc_constraints(2,1))+"<=BETA<=" +@str(dcc_constraints(2,2))
			endif
		else
			temp_out(!place-15,1)="Initial Values: ALPHA=" +@str(dcc_param(1)) +", BETA=" +@str(dcc_param(2)) +" with restrictions: " +@str(dcc_constraints(2,1))+"<=BETA<=" +@str(dcc_constraints(2,2))
		endif
			if !hess_approx=1 then
				if !rob_err=1 then
					temp_out(!place-13,1)="Coefficient covariance computed using observed Hessian and the delta method"
				else
					temp_out(!place-13,1)="Coefficient covariance computed using sandwich method (robust Huber-White) and the delta method"
				endif				
			else
				if !hess_approx=2 then
					if !rob_err=1 then
						temp_out(!place-13,1)="Coefficient covariance computed using outer product of gradients and the delta method"
					else
						temp_out(!place-13,1)="Coefficient covariance computed using sandwich method (robust Huber-White) and the delta method"
					endif
				else
					if !rob_err=1 then
						temp_out(!place-13,1)="Coefficient covariance computed using numerical approximation and the delta method"
					else
						temp_out(!place-13,1)="Coefficient covariance computed using sandwich method (robust Huber-White) and the delta method"
					endif
				endif
			endif 
		endif
	endif
endif

if !grid_search=1 then
	temp_out(!place-15,1)=temp_out(!place-15,1)+" with grid search"
endif

endsub

' ============================================================================================================================
' logit transformation of the starting parameters of the DCC model

subroutine dcc_param_trans(vector start_param, matrix dcc_constraints, svector dcc_settings)

!tdist=@val(dcc_settings(4))
!tdist_dcc=@val(dcc_settings(5))
!cnst_dcc=@val(dcc_settings(17))
!cnst_dcc_t=@val(dcc_settings(18))

if !tdist_dcc=1 then
	vector(3) dcc_coefs
else
	vector(2) dcc_coefs
endif

dcc_coefs=start_param

if !cnst_dcc=0 then
	if !tdist_dcc=1 then
		if !cnst_dcc_t=1 then
			dcc_coefs(3)=@log((start_param(3)-dcc_constraints(3,1))/(dcc_constraints(3,2)-start_param(3)))
		endif
	endif
else
	if !cnst_dcc=1 then
		dcc_coefs(1)=@log((start_param(1)-dcc_constraints(1,1))/(dcc_constraints(1,2)-start_param(1)))
		dcc_coefs(2)=@log((start_param(2)-dcc_constraints(2,1))/(dcc_constraints(2,2)-start_param(2)))
		if !tdist_dcc=1 then
			if !cnst_dcc_t=1 then
				dcc_coefs(3)=@log((start_param(3)-dcc_constraints(3,1))/(dcc_constraints(3,2)-start_param(3)))
			endif
		endif
	else
		if !cnst_dcc=2 then
			dcc_coefs(1)=@log((start_param(1)-dcc_constraints(1,1))/(dcc_constraints(1,2)-start_param(1)))
			if !tdist_dcc=1 then
				if !cnst_dcc_t=1 then
					dcc_coefs(3)=@log((start_param(3)-dcc_constraints(3,1))/(dcc_constraints(3,2)-start_param(3)))
				endif
			endif
		else
			if !cnst_dcc=3 then
				dcc_coefs(2)=@log((start_param(2)-dcc_constraints(2,1))/(dcc_constraints(2,2)-start_param(2)))
				if !tdist_dcc=1 then
					if !cnst_dcc_t=1 then
						dcc_coefs(3)=@log((start_param(3)-dcc_constraints(3,1))/(dcc_constraints(3,2)-start_param(3)))
					endif
				endif
			else
			endif
		endif
	endif
endif

endsub

' ============================================================================================================================
' parameters estimation of the DCC model with the maximum likelihood method

subroutine dcc_estim(vector dcc_coefs, matrix et_mat, matrix dcc_param_constraints, svector dcc_settings)

vector(num_obs) dcc_logl_vec

!rob_err=@val(dcc_settings(6))
!grid_search=@val(dcc_settings(7))
!dim_grid=@val(dcc_settings(19))
!tol=@val(dcc_settings(8))
%hess_appro=dcc_settings(11)
%step_method=dcc_settings(13)
!max_ite=@val(dcc_settings(9))
!trust_val=@val(dcc_settings(10))
!cnst_dcc=@val(dcc_settings(17))
!radio_tdist_dcc=@val(dcc_settings(5))
!cnst_dcc_t=@val(dcc_settings(18))

if !rob_err=1 then
	tic
	if !grid_search=1 then
		
		table dcc_grid_errs

		dcc_grid_errs(1,1)="Grid search for the DCC model"
		dcc_grid_errs(2,1)="Number of the starting values"
		dcc_grid_errs(2,2)="ALPHA starting parameter"
		dcc_grid_errs(2,3)="BETA starting parameter"
		dcc_grid_errs(2,5)="Optimization message"
		dcc_grid_errs(2,6)="ALPHA coefficient"
		dcc_grid_errs(2,7)="BETA coefficient"
		dcc_grid_errs(2,9)="The logarithm of the likelihood function"

		if !radio_tdist_dcc=1 then
			dcc_grid_errs(2,4)="The degrees of freedom starting parameter"
			dcc_grid_errs(2,8)="The degrees of freedom coefficient"
		endif

		%err_dcc="Error"
		!grid_c=1

		while %err_dcc="Error"

			statusline Grid search for the parameters estimation of the DCC model. Number of the starting values initialized: !grid_c

			if !grid_c>!dim_grid then
				temp_out(!place-14,1)=@optmessage
				@uiprompt("Error: all the staring values exausted for the grid search. The optimization algorithm is unable to converge."  + @chr(13) + "Details in the DCC_GRID_ERRS table.", "O")
				stop
			else
				call dcc_param_trans(@transpose(@rowextract(dcc_grid_start, !grid_c)), dcc_param_constraints, dcc_settings)
			endif
			
			dcc_grid_errs(2+!grid_c,1)=!grid_c
			dcc_grid_errs(2+!grid_c,2)=dcc_grid_start(!grid_c, 1)
			dcc_grid_errs(2+!grid_c,3)=dcc_grid_start(!grid_c, 2)

				if !radio_tdist_dcc=1 then
					dcc_grid_errs(2+!grid_c,4)=dcc_grid_start(!grid_c, 3)
				endif

			optimize(ml=1, c=!tol, deriv=high, finalh=dcc_hessian, hess=%hess_appro, step=%step_method, m=!max_ite, trust=!trust_val, noerr) loglike_dcc(dcc_logl_vec, dcc_coefs, et_mat, !cnst_dcc, !radio_tdist_dcc, !cnst_dcc_t, dcc_param_constraints, !grid_search)

			if @optstatus<>0 then
				%err_dcc="Error"
			else
				if @optstatus=0 then
					%err_dcc="True"
				endif
			endif

			vector dcc_coefs_un=dcc_coefs
			call dcc_untransform(dcc_coefs_un, dcc_param_constraints, dcc_settings)

			dcc_grid_errs(2+!grid_c, 5)=@optmessage	
			dcc_grid_errs(2+!grid_c,6)=dcc_coefs_un(1)
			dcc_grid_errs(2+!grid_c,7)=dcc_coefs_un(2)
			dcc_grid_errs(2+!grid_c,9)=@sum(dcc_logl_vec)
			
			if !radio_tdist_dcc=1 then
				dcc_grid_errs(2+!grid_c, 8)=dcc_coefs_un(3)
			endif

			!grid_c=!grid_c+1
		wend
			temp_out(!place-15,1)=temp_out(!place-15,1)+" (Number of tested starting values: " +@str(!grid_c-1)+")"
	else

		statusline Parameter estimation of the DCC model
		optimize(ml=1, c=!tol, deriv=high, finalh=dcc_hessian, hess=%hess_appro, step=%step_method, m=!max_ite, trust=!trust_val) loglike_dcc(dcc_logl_vec, dcc_coefs, et_mat, !cnst_dcc, !radio_tdist_dcc, !cnst_dcc_t, dcc_param_constraints, !grid_search) 

	endif

	!elapsed=@toc

	vector dcc_mlse=@sqrt(@getmaindiagonal(-@inverse(dcc_hessian)))								' standard errors based on Hessian matrix
	matrix dcc_covar=-@inverse(dcc_hessian)																		' matrix with the  coefficients covariances
else
	tic

	if !grid_search=1 then

		table dcc_grid_errs

		dcc_grid_errs(2,1)="Number of the starting values"
		dcc_grid_errs(2,2)="ALPHA starting parameter"
		dcc_grid_errs(2,3)="BETA starting parameter"
		dcc_grid_errs(2,5)="Optimization message"
		dcc_grid_errs(2,6)="ALPHA coefficient"
		dcc_grid_errs(2,7)="BETA coefficient"
		dcc_grid_errs(2,9)="The logarithm of the likelihood function"

		if !radio_tdist_dcc=1 then
			dcc_grid_errs(2,4)="The degrees of freedom starting parameter"
			dcc_grid_errs(2,8)="The degrees of freedom coefficient"
		endif

		%err_dcc="Error"
		!grid_c=1

		while %err_dcc="Error"

			statusline Grid search for the parameters estimation of the DCC model. Number of the starting values initialized: !grid_c

			if !grid_c>!dim_grid then
				temp_out(!place-14,1)=@optmessage
				@uiprompt("Error: all the staring values exausted for the grid search. The optimization algorithm is unable to converge."  + @chr(13) + "Details in the DCC_GRID_ERRS table.", "O")
				stop
			else
				call dcc_param_trans(@transpose(@rowextract(dcc_grid_start, !grid_c)), dcc_param_constraints, dcc_settings)
			endif

			dcc_grid_errs(2+!grid_c,1)=!grid_c
			dcc_grid_errs(2+!grid_c,2)=dcc_grid_start(!grid_c, 1)
			dcc_grid_errs(2+!grid_c,3)=dcc_grid_start(!grid_c, 2)

			if !radio_tdist_dcc=1 then
				dcc_grid_errs(2+!grid_c,4)=dcc_grid_start(!grid_c, 3)
			endif

			!hess_approx=2																									' OPG Hessian matrix
			%hess_appro="OPG"
			optimize(ml=1, c=!tol, deriv=high, finalh=dcc_hessian, hess=%hess_appro, step=%step_method, m=!max_ite, trust=!trust_val, noerr) loglike_dcc(dcc_logl_vec, dcc_coefs, et_mat, !cnst_dcc, !radio_tdist_dcc, !cnst_dcc_t, dcc_param_constraints, !grid_search)

			%status=@optmessage
			statusline {%status}

			if @optstatus<>0 then
				%err_dcc="Error"
			else
				if @optstatus=0 then
					%err_dcc="True"
				endif
			endif

			vector dcc_coefs_un=dcc_coefs
			call dcc_untransform(dcc_coefs_un, dcc_param_constraints, dcc_settings)

			dcc_grid_errs(2+!grid_c, 5)=@optmessage	
			dcc_grid_errs(2+!grid_c, 6)=dcc_coefs_un(1)
			dcc_grid_errs(2+!grid_c, 7)=dcc_coefs_un(2)
			dcc_grid_errs(2+!grid_c, 9)=@sum(dcc_logl_vec)

			if !radio_tdist_dcc=1 then
				dcc_grid_errs(2+!grid_c, 8)=dcc_coefs_un(3)
			endif

			!grid_c=!grid_c+1
		wend
			temp_out(!place-15,1)=temp_out(!place-15,1)+" (Number of tested starting values: " +@str(!grid_c-1)+")"

		matrix dcc_hess_opg=dcc_hessian

		!hess_approx=3																										' numeric Hessian matrix
		%hess_appro="Numeric"
		optimize(ml=1, c=!tol, deriv=high, finalh=dcc_hessian, hess=%hess_appro, step=%step_method, m=!max_ite, trust=!trust_val) loglike_dcc(dcc_logl_vec, dcc_coefs, et_mat, !cnst_dcc, !radio_tdist_dcc, !cnst_dcc_t, dcc_param_constraints, !grid_search)
		matrix dcc_hess_num=dcc_hessian
		!elapsed=@toc
		matrix dcc_covar=-@inverse(dcc_hess_num)*(-dcc_hess_opg)*(-@inverse(dcc_hess_num))	' robust covariance matrix Huber-White
		vector dcc_mlse=@sqrt(@getmaindiagonal(dcc_covar))												' robust standard errors Huber-White

	else

		statusline Parameter estimation of the DCC model

		!hess_approx=2
		%hess_appro="OPG"																								' OPG Hessian matrix
		optimize(ml=1, c=!tol, deriv=high, finalh=dcc_hessian, hess=%hess_appro, step=%step_method, m=!max_ite, trust=!trust_val) loglike_dcc(dcc_logl_vec, dcc_coefs, et_mat, !cnst_dcc, !radio_tdist_dcc, !cnst_dcc_t, dcc_param_constraints, !grid_search)
		matrix dcc_hess_opg=dcc_hessian
		%status=@optmessage
		statusline {%status}

		!hess_approx=3																										' numeric Hessian matrix
		%hess_appro="Numeric"
		optimize(ml=1, c=!tol, deriv=high, finalh=dcc_hessian, hess=%hess_appro, step=%step_method, m=!max_ite, trust=!trust_val) loglike_dcc(dcc_logl_vec, dcc_coefs, et_mat, !cnst_dcc, !radio_tdist_dcc, !cnst_dcc_t, dcc_param_constraints, !grid_search)
		matrix dcc_hess_num=dcc_hessian
		!elapsed=@toc
		matrix dcc_covar=-@inverse(dcc_hess_num)*(-dcc_hess_opg)*(-@inverse(dcc_hess_num))	' robust covariance matrix Huber-White
		vector dcc_mlse=@sqrt(@getmaindiagonal(dcc_covar))												' robust standard errors Huber-White
	endif
endif

call del_obj("dcc_logl")

series dcc_logl
mtos(dcc_logl_vec, dcc_logl, dcc_smpl)

if !rob_err=1 then
	%status=@optmessage
	statusline {%status}
endif

endsub

subroutine local dcc_untransform(vector dcc_coefs, matrix dcc_constraints, svector dcc_settings)

!cnst_dcc=@val(dcc_settings(17))
!radio_tdist_dcc=@val(dcc_settings(5))
!cnst_dcc_t=@val(dcc_settings(18))

!lb_a=dcc_constraints(1,1)
!ub_a=dcc_constraints(1,2)
!lb_b=dcc_constraints(2,1)
!ub_b=dcc_constraints(2,2)
!lb_v=dcc_constraints(3,1)
!ub_v=dcc_constraints(3,2)

vector dcc_coefs_cnstr=dcc_coefs

if !cnst_dcc=1 then
	dcc_coefs(1)=(!lb_a+(!ub_a-!lb_a)*@exp(dcc_coefs_cnstr(1))/(1+@exp(dcc_coefs_cnstr(1))))
	dcc_coefs(2)=(!lb_b+(!ub_b-!lb_b)*@exp(dcc_coefs_cnstr(2))/(1+@exp(dcc_coefs_cnstr(2))))
	if !radio_tdist_dcc=1 and !cnst_dcc_t=1 then
		dcc_coefs(3)=(!lb_v+(!ub_v-!lb_v)*@exp(dcc_coefs_cnstr(3))/(1+@exp(dcc_coefs_cnstr(3))))
	endif
else
	if !cnst_dcc=2 then
		dcc_coefs(1)=(!lb_a+(!ub_a-!lb_a)*@exp(dcc_coefs_cnstr(1))/(1+@exp(dcc_coefs_cnstr(1))))
		if !radio_tdist_dcc=1 and !cnst_dcc_t=1 then
			dcc_coefs(3)=(!lb_v+(!ub_v-!lb_v)*@exp(dcc_coefs_cnstr(3))/(1+@exp(dcc_coefs_cnstr(3))))
		endif
	else
		if !cnst_dcc=3 then
			dcc_coefs(2)=(!lb_b+(!ub_b-!lb_b)*@exp(dcc_coefs_cnstr(2))/(1+@exp(dcc_coefs_cnstr(2))))
			if !radio_tdist_dcc=1 and !cnst_dcc_t=1 then
				dcc_coefs(3)=(!lb_v+(!ub_v-!lb_v)*@exp(dcc_coefs_cnstr(3))/(1+@exp(dcc_coefs_cnstr(3))))
			endif
		endif
	endif
endif
if !radio_tdist_dcc=1 and !cnst_dcc_t=1 then
	dcc_coefs(3)=(!lb_v+(!ub_v-!lb_v)*@exp(dcc_coefs_cnstr(3))/(1+@exp(dcc_coefs_cnstr(3))))
endif


endsub

' ============================================================================================================================
' errors of coefficients based on the delta method and undoing transformation of the coefficients
' vectors with z-statistics and p-values

subroutine local dcc_errors(vector dcc_coefs, vector dcc_mlse, vector dcc_zstat, vector dcc_pval, matrix dcc_covar, matrix dcc_constraints, svector dcc_settings)

!cnst_dcc=@val(dcc_settings(17))
!radio_tdist_dcc=@val(dcc_settings(5))
!cnst_dcc_t=@val(dcc_settings(18))

!lb_a=dcc_constraints(1,1)
!ub_a=dcc_constraints(1,2)
!lb_b=dcc_constraints(2,1)
!ub_b=dcc_constraints(2,2)
!lb_v=dcc_constraints(3,1)
!ub_v=dcc_constraints(3,2)

vector dcc_coefs_cnstr=dcc_coefs
matrix dcc_covar_cnstr=dcc_covar

if !cnst_dcc=1 then

	!a_res=(!ub_a-!lb_a)*@exp(dcc_coefs(1))/(1+@exp(dcc_coefs(1)))^2
	!b_res=(!ub_b-!lb_b)*@exp(dcc_coefs(2))/(1+@exp(dcc_coefs(2)))^2

	dcc_covar(1,1)=!a_res*dcc_covar_cnstr(1,1)*!a_res
	dcc_covar(1,2)=!b_res*dcc_covar_cnstr(1,2)*!a_res
	dcc_covar(2,1)=!b_res*dcc_covar_cnstr(2,1)*!a_res
	dcc_covar(2,2)=!b_res*dcc_covar_cnstr(2,2)*!b_res

	dcc_coefs(1)=(!lb_a+(!ub_a-!lb_a)*@exp(dcc_coefs_cnstr(1))/(1+@exp(dcc_coefs_cnstr(1))))
	dcc_coefs(2)=(!lb_b+(!ub_b-!lb_b)*@exp(dcc_coefs_cnstr(2))/(1+@exp(dcc_coefs_cnstr(2))))

	dcc_mlse(1)=@sqrt(dcc_covar(1,1))
	dcc_mlse(2)=@sqrt(dcc_covar(2,2))

	if !radio_tdist_dcc=1 and !cnst_dcc_t=1 then
		!v_res=(!ub_v-!lb_v)*@exp(dcc_coefs(3))/(1+@exp(dcc_coefs(3)))^2
		dcc_covar(3,3)=!v_res*dcc_covar_cnstr(3,3)*!v_res
		dcc_covar(1,3)=!a_res*dcc_covar_cnstr(1,3)*!v_res
		dcc_covar(3,1)=!v_res*dcc_covar_cnstr(1,3)*!a_res
		dcc_covar(2,3)=!b_res*dcc_covar_cnstr(2,3)*!v_res
		dcc_covar(3,2)=!v_res*dcc_covar_cnstr(2,3)*!b_res
		dcc_coefs(3)=(!lb_v+(!ub_v-!lb_v)*@exp(dcc_coefs_cnstr(3))/(1+@exp(dcc_coefs_cnstr(3))))
		dcc_mlse(3)=@sqrt(dcc_covar(3,3))
	endif
else
	if !cnst_dcc=2 then

		!a_res=(!ub_a-!lb_a)*@exp(dcc_coefs(1))/(1+@exp(dcc_coefs(1)))^2
	
		dcc_covar(1,1)=!a_res*dcc_covar_cnstr(1,1)*!a_res
		dcc_covar(1,2)=1*dcc_covar_cnstr(1,2)*!a_res
		dcc_covar(2,1)=1*dcc_covar_cnstr(2,1)*!a_res
		dcc_covar(2,2)=dcc_covar_cnstr(2,2)

		dcc_coefs(1)=(!lb_a+(!ub_a-!lb_a)*@exp(dcc_coefs_cnstr(1))/(1+@exp(dcc_coefs_cnstr(1))))

		dcc_mlse(1)=@sqrt(dcc_covar(1,1))
	
		if !radio_tdist_dcc=1 and !cnst_dcc_t=1 then
			!v_res=(!ub_v-!lb_v)*@exp(dcc_coefs(3))/(1+@exp(dcc_coefs(3)))^2
			dcc_covar(3,3)=!v_res*dcc_covar_cnstr(3,3)*!v_res
			dcc_covar(1,3)=!a_res*dcc_covar_cnstr(1,3)*!v_res
			dcc_covar(3,1)=!v_res*dcc_covar_cnstr(1,3)*!a_res
			dcc_covar(2,3)=1*dcc_covar_cnstr(2,3)*!v_res
			dcc_covar(3,2)=!v_res*dcc_covar_cnstr(2,3)*1
			dcc_coefs(3)=(!lb_v+(!ub_v-!lb_v)*@exp(dcc_coefs_cnstr(3))/(1+@exp(dcc_coefs_cnstr(3))))
			dcc_mlse(3)=@sqrt(dcc_covar(3,3))
		endif
	else
		if !cnst_dcc=3 then

			!b_res=(!ub_b-!lb_b)*@exp(dcc_coefs(2))/(1+@exp(dcc_coefs(2)))^2

			dcc_covar(1,1)=dcc_covar_cnstr(1,1)
			dcc_covar(1,2)=!b_res*dcc_covar_cnstr(1,2)*1
			dcc_covar(2,1)=!b_res*dcc_covar_cnstr(2,1)*1
			dcc_covar(2,2)=!b_res*dcc_covar_cnstr(2,2)*!b_res

			dcc_coefs(2)=(!lb_b+(!ub_b-!lb_b)*@exp(dcc_coefs_cnstr(2))/(1+@exp(dcc_coefs_cnstr(2))))
	
			dcc_mlse(2)=@sqrt(dcc_covar(2,2))

			if !radio_tdist_dcc=1 and !cnst_dcc_t=1 then
				!v_res=(!ub_v-!lb_v)*@exp(dcc_coefs(3))/(1+@exp(dcc_coefs(3)))^2
				dcc_covar(3,3)=!v_res*dcc_covar_cnstr(3,3)*!v_res
				dcc_covar(1,3)=1*dcc_covar_cnstr(1,3)*!v_res
				dcc_covar(3,1)=!v_res*dcc_covar_cnstr(1,3)*1
				dcc_covar(2,3)=!b_res*dcc_covar_cnstr(2,3)*!v_res
				dcc_covar(3,2)=!v_res*dcc_covar_cnstr(2,3)*!b_res
				dcc_coefs(3)=(!lb_v+(!ub_v-!lb_v)*@exp(dcc_coefs_cnstr(3))/(1+@exp(dcc_coefs_cnstr(3))))
				dcc_mlse(3)=@sqrt(dcc_covar(3,3))
			endif
		endif
	endif
endif

if !radio_tdist_dcc=1 and !cnst_dcc_t=1 then

	!v_res=(!ub_v-!lb_v)*@exp(dcc_coefs(3))/(1+@exp(dcc_coefs(3)))^2
	dcc_covar(3,3)=!v_res*dcc_covar_cnstr(3,3)*!v_res
	dcc_covar(1,3)=1*dcc_covar_cnstr(1,3)*!v_res
	dcc_covar(3,1)=!v_res*dcc_covar_cnstr(1,3)*1
	dcc_covar(2,3)=1*dcc_covar_cnstr(2,3)*!v_res
	dcc_covar(3,2)=!v_res*dcc_covar_cnstr(2,3)*1
	dcc_coefs(3)=(!lb_v+(!ub_v-!lb_v)*@exp(dcc_coefs_cnstr(3))/(1+@exp(dcc_coefs_cnstr(3))))
	dcc_mlse(3)=@sqrt(dcc_covar(3,3))
endif

vector dcc_zstat=@ediv(dcc_coefs, dcc_mlse)
if !radio_tdist_dcc=1 then
	vector(3) dcc_one=1
else
	vector(2) dcc_one=1
endif

vector dcc_pval=2*(dcc_one-@cnorm(@abs(dcc_zstat)))

endsub

' ============================================================================================================================
' obtaining the conditional correlations and the conditional covariances

subroutine dcc_forecasts(vector dcc_coefs, matrix et_mat, matrix ht_mat, svector dcc_settings)

statusline Post-estimation computations i.e. errors, fitted values, forecasts...

matrix cov_unc=@covp(et_mat)
matrix cor_unc=@cor(et_mat)

!model=@val(dcc_settings(1))
!num_var=@val(dcc_settings(22))
num_obs=@val(dcc_settings(15))

' temporary matrices with the correlations and covariances
for !i5=1 to !num_var
	for !j5=!i5+1 to !num_var
	vector(num_obs) temp_rt_{!i5}{!j5}
	vector(num_obs) temp_cov_{!i5}{!j5}
		if !model=1 then
			call del_obj("rgarch_cor_"+@str(!i5)+"_"+@str(!j5))
			series rgarch_cor_{!i5}_{!j5}
			call del_obj("rgarch_cov_"+@str(!i5)+"_"+@str(!j5))
			series rgarch_cov_{!i5}_{!j5}
		else
			if !model=2 then
				call del_obj("garch_cor_"+@str(!i5)+"_"+@str(!j5))
				series garch_cor_{!i5}_{!j5}	
				call del_obj("garch_cov_"+@str(!i5)+"_"+@str(!j5))
				series garch_cov_{!i5}_{!j5}
			else
				call del_obj("ev_garch_cor_"+@str(!i5)+"_"+@str(!j5))
				series ev_garch_cor_{!i5}_{!j5}
				call del_obj("ev_garch_cov_"+@str(!i5)+"_"+@str(!j5))
				series ev_garch_cov_{!i5}_{!j5}
			endif
		endif
	next
next

' initial values of the correlations and covariances
for !i5=1 to !num_var
	for !j5=!i5+1 to !num_var
		temp_rt_{!i5}{!j5}(1)=cor_unc(!i5,!j5)
		temp_cov_{!i5}{!j5}(1)=cov_unc(!i5,!j5)
	next
next

' obtaining in-the-sample conditional correlations and conditional covariances
sym(!num_var) dcc_Q=@covp(et_mat)

for !i4=2 to num_obs
	sym dcc_Q=cor_unc*(1-dcc_coefs(1)-dcc_coefs(2))+dcc_coefs(1)*(@transpose(@rowextract(et_mat,!i4-1))*@rowextract(et_mat,!i4-1))+dcc_coefs(2)*dcc_Q
	sym dcc_Qs = @inverse(@sqrt(@makediagonal(@getmaindiagonal(dcc_Q))))
	sym dcc_R=@qform(dcc_Q, dcc_Qs)
	sym dcc_D=@sqrt(@makediagonal(@rowextract(ht_mat,!i4)))
	matrix dcc_H=@qform(dcc_R, dcc_D)
	for !i5=1 to !num_var
		for !j5=!i5+1 to !num_var
			temp_rt_{!i5}{!j5}(!i4)=dcc_R(!i5,!j5)
			temp_cov_{!i5}{!j5}(!i4)=dcc_H(!i5,!j5)
		next
	next
next

' transformation of the matrices with the conditional correlations and covariances into series

for !i5=1 to !num_var
	for !j5=!i5+1 to !num_var
		if !model=1 then
			mtos(temp_rt_{!i5}{!j5}, rgarch_cor_{!i5}_{!j5})
			mtos(temp_cov_{!i5}{!j5}, rgarch_cov_{!i5}_{!j5})
		else
			if !model=2 then
				mtos(temp_rt_{!i5}{!j5}, garch_cor_{!i5}_{!j5})
				mtos(temp_cov_{!i5}{!j5}, garch_cov_{!i5}_{!j5})
			else
				mtos(temp_rt_{!i5}{!j5}, ev_garch_cor_{!i5}_{!j5}, dcc_smpl)
				mtos(temp_cov_{!i5}{!j5}, ev_garch_cov_{!i5}_{!j5}, dcc_smpl)
			endif
		endif
	next
next

' one-period ahead forecasts of the conditional correlations and the conditional covarinaces

sym dcc_Qf=cor_unc*(1-dcc_coefs(1)-dcc_coefs(2))+dcc_coefs(1)*(@transpose(@rowextract(et_mat,num_obs))*@rowextract(et_mat,num_obs))+dcc_coefs(2)*dcc_Q
sym dcc_Qsf = @inverse(@sqrt(@makediagonal(@getmaindiagonal(dcc_Qf))))
sym dcc_R_fcast=@qform(dcc_Qf, dcc_Qsf)

if !model=1 then
	if rgarch_fcast_ht>0 then
		sym dcc_Df=@sqrt(@makediagonal(@transpose(rgarch_fcast_ht)))
		matrix dcc_Ht_fcast=@qform(dcc_R_fcast, dcc_Df)
	else
		@uiprompt("Error: at least one of the forecasts of the conditional variances is negative")
	endif
else
		if !model=2 then
			if garch_fcast_ht>0 then
				sym dcc_Df=@sqrt(@makediagonal(@transpose(garch_fcast_ht)))
				matrix dcc_Ht_fcast=@qform(dcc_R_fcast, dcc_Df)
			else
				@uiprompt("Error: at least one of the forecasts of the conditional variances is negative")
			endif
		else
			if !model=3 then
				if ev_garch_fcast_ht>0 then
					sym dcc_Df=@sqrt(@makediagonal(@transpose(ev_garch_fcast_ht)))
					matrix dcc_Ht_fcast=@qform(dcc_R_fcast, dcc_Df)
				else
					@uiprompt("Error: at least one of the forecasts of the conditional variances is negative")
				endif
			endif
		endif
endif

endsub

' ============================================================================================================================
' place the final information into the temporary output table
' obtaining additinal measures, i.e. sum of the logarithm of likelihood, mean of the logarithm of likelihood, Akaike information criterion, Bayesian Schwarz information criterion, Hannan-Quinn information criterion

subroutine dcc_output2(vector dcc_coefs, scalar num_obs, series dcc_logl, vector dcc_mlse, vector dcc_zstat, vector dcc_pval, scalar tdist_dcc)

' information criteria for the DCC model
!dcc_aic=-2*@sum(dcc_logl)/num_obs+2*(2/num_obs)
!dcc_bic=-2*@sum(dcc_logl)/num_obs+2*(@log(num_obs)/num_obs)
!dcc_hqc=-2*@sum(dcc_logl)/num_obs+2*2*(@log(@log(num_obs))/num_obs)

vector(7) dcc_astat

dcc_astat(1)=@sum(dcc_logl)				' sum of the logarithm of likelihood
dcc_astat(2)=@mean(dcc_logl)				' mean of the logarithm of likelihood
dcc_astat(3)=@rows(dcc_coefs)			' the number of the coefficients
dcc_astat(4)=!dcc_aic							' Akaike information criterion 
dcc_astat(5)=!dcc_bic							' Bayesian Schwarz criterion
dcc_astat(6)=!dcc_hqc							' Hannan-Quinn criterion
dcc_astat(7)=num_obs						' the number of observation

temp_out(!place-14,1)=%status
temp_out.setlines(!place-12,1) +d
temp_out.setlines(!place-12,2) +d
temp_out.setlines(!place-12,3) +d
temp_out.setlines(!place-12,4) +d
temp_out.setlines(!place-12,5) +d
temp_out(!place-11,2)="Coefficient"
temp_out(!place-11,3)="Std. Error"
temp_out(!place-11,4)="z-Statistic"
temp_out.setjust(!place-11,2) right
temp_out.setjust(!place-11,3) right
temp_out.setjust(!place-11,4) right
temp_out(!place-11,5)="Prob."
temp_out.setjust(!place-11,5) right
temp_out.setindent(!place-11,5) 5
temp_out.setlines(!place-10,1) +d
temp_out.setlines(!place-10,2) +d
temp_out.setlines(!place-10,3) +d
temp_out.setlines(!place-10,4) +d
temp_out.setlines(!place-10,5) +d

temp_out(!place-9,1)="ALPHA"
temp_out(!place-9,2)=dcc_coefs(1)
temp_out(!place-9,3)=dcc_mlse(1)
temp_out(!place-9,4)=dcc_zstat(1)
temp_out(!place-9,5)=dcc_pval(1)

temp_out(!place-8,1)="BETA"
temp_out(!place-8,2)=dcc_coefs(2)
temp_out(!place-8,3)=dcc_mlse(2)
temp_out(!place-8,4)=dcc_zstat(2)
temp_out(!place-8,5)=dcc_pval(2)

temp_out.setjust(!place-9,2) right
temp_out.setjust(!place-9,3) right
temp_out.setjust(!place-9,4) right
temp_out.setjust(!place-9,5) right
temp_out.setjust(!place-8,2) right
temp_out.setjust(!place-8,3) right
temp_out.setjust(!place-8,4) right
temp_out.setjust(!place-8,5) right

	if tdist_dcc=2 then

		temp_out.setlines(!place-7,1) +d
		temp_out.setlines(!place-7,2) +d
		temp_out.setlines(!place-7,3) +d
		temp_out.setlines(!place-7,4) +d
		temp_out.setlines(!place-7,5) +d
		temp_out(!place-6,1)="Log likelihood"
		temp_out(!place-5,1)="Avg. log likelihood"
		temp_out(!place-4,1)="Number of Coefs"
		temp_out.setjust(!place-6,1) left
		temp_out.setjust(!place-5,1) left
		temp_out.setjust(!place-4,1) left
		temp_out.setjust(!place-1,1) left
		temp_out(!place-6,2)=@sum(dcc_logl)
		temp_out(!place-5,2)=@mean(dcc_logl)
		temp_out.setjust(!place-6,2) right
		temp_out.setjust(!place-5,2) right
		temp_out(!place-4,2)=@str(2)
		temp_out.setjust(!place-4,2) right
		temp_out(!place-6,3)="Akaike info criterion"
		temp_out(!place-5,3)="Schwarz criterion"
		temp_out(!place-4,3)="Hannan-Quinn criter."
		temp_out.setindent(!place-6,3) 10
		temp_out.setindent(!place-5,3) 10
		temp_out.setindent(!place-4,3) 10
		temp_out(!place-6,5)=!dcc_aic
		temp_out(!place-5,5)=!dcc_bic
		temp_out(!place-4,5)=!dcc_hqc
		temp_out.setjust(!place-6,5) right
		temp_out.setjust(!place-5,5) right
		temp_out.setjust(!place-4,5) right
		temp_out.setlines(!place-3,1) +d
		temp_out.setlines(!place-3,2) +d
		temp_out.setlines(!place-3,3) +d
		temp_out.setlines(!place-3,4) +d
		temp_out.setlines(!place-3,5) +d
		temp_out(!place-2,1)="Elapsed time: " +@str(!elapsed) + " seconds (or "+@str(!elapsed/60)+" minutes)"
		temp_out(!place-1,1)="Completed successfully"

	else

		temp_out(!place-7,1)="T-DIST. DOF"
		temp_out(!place-7,2)=dcc_coefs(3)
		temp_out(!place-7,3)=dcc_mlse(3)
		temp_out(!place-7,4)=dcc_zstat(3)
		temp_out(!place-7,5)=dcc_pval(3)
		temp_out.setjust(!place-7,2) right
		temp_out.setjust(!place-7,3) right
		temp_out.setjust(!place-7,4) right
		temp_out.setjust(!place-7,5) right
		temp_out.setlines(!place-6,1) +d
		temp_out.setlines(!place-6,2) +d
		temp_out.setlines(!place-6,3) +d
		temp_out.setlines(!place-6,4) +d
		temp_out.setlines(!place-6,5) +d
		temp_out(!place-5,1)="Log likelihood"
		temp_out(!place-4,1)="Avg. log likelihood"
		temp_out(!place-3,1)="Number of Coefs"
		temp_out.setjust(!place-5,1) left
		temp_out.setjust(!place-4,1) left
		temp_out.setjust(!place-3,1) left
		temp_out.setjust(!place,1) left
		temp_out(!place-5,2)=@sum(dcc_logl)
		temp_out(!place-4,2)=@mean(dcc_logl)
		temp_out.setjust(!place-5,2) right
		temp_out.setjust(!place-4,2) right
		temp_out(!place-3,2)=@str(3)
		temp_out.setjust(!place-3,2) right
		temp_out(!place-5,3)="Akaike info criterion"
		temp_out(!place-4,3)="Schwarz criterion"
		temp_out(!place-3,3)="Hannan-Quinn criter."
		temp_out.setindent(!place-5,3) 10
		temp_out.setindent(!place-4,3) 10
		temp_out.setindent(!place-3,3) 10
		temp_out(!place-5,5)=!dcc_aic
		temp_out(!place-4,5)=!dcc_bic
		temp_out(!place-3,5)=!dcc_hqc
		temp_out.setjust(!place-5,5) right
		temp_out.setjust(!place-4,5) right
		temp_out.setjust(!place-3,5) right
		temp_out.setlines(!place-2,1) +d
		temp_out.setlines(!place-2,2) +d
		temp_out.setlines(!place-2,3) +d
		temp_out.setlines(!place-2,4) +d
		temp_out.setlines(!place-2,5) +d
		temp_out(!place-1,1)="Elapsed time: " +@str(!elapsed) + " seconds (or "+@str(!elapsed/60)+" minutes)"
		temp_out(!place,1)="Completed successfully"

	endif

endsub

'====================================================================================================================
'create final table with outputs

subroutine print_final_table(scalar model, scalar place, scalar action)

if model=1 then
	%tab_name="dcc_rgarch"
	if action=1 then									' overwrite output table
		if @isobject(%tab_name) then
			delete {%tab_name}
			table {%tab_name}
			tabplace({%tab_name}, temp_out,  "A1", "A1", "H"+@str(place+30))
		else
			table {%tab_name}
			tabplace({%tab_name}, temp_out,  "A1", "A1", "H"+@str(place+30))
		endif	
	else													' create a new output table with next available name
		if @isobject(%tab_name) then
			%tab_name=@getnextname(%tab_name)
			table {%tab_name}
			tabplace({%tab_name}, temp_out,  "A1", "A1", "H"+@str(place+30))
		else
			table {%tab_name}
			tabplace({%tab_name}, temp_out,  "A1", "A1", "H"+@str(place+30))
		endif
	endif
else
	if model=2 then
		%tab_name="dcc_garch"
		if action=1 then								' overwrite output table
			if @isobject(%tab_name) then
				delete %tab_name
				table {%tab_name}
				tabplace({%tab_name}, temp_out,  "A1", "A1", "H"+@str(place+30))
			else
				table {%tab_name}
				tabplace({%tab_name}, temp_out,  "A1", "A1", "H"+@str(place+30))
			endif
		else												' create a new output table with next available name
			if @isobject(%tab_name) then
				%tab_name=@getnextname(%tab_name)
				table {%tab_name}
				tabplace({%tab_name}, temp_out,  "A1", "A1", "H"+@str(place+30))
			else
				table {%tab_name}
				tabplace({%tab_name}, temp_out,  "A1", "A1", "H"+@str(place+30))
			endif
		endif
	else
		if model=3 then	
			%tab_name="dcc_ev_garch"
			if action=1 then								' overwrite output table
				if @isobject(%tab_name) then
					delete %tab_name
					table {%tab_name}
					tabplace({%tab_name}, temp_out,  "A1", "A1", "H"+@str(!place+30))
				else
					table {%tab_name}
					tabplace({%tab_name}, temp_out,  "A1", "A1", "H"+@str(!place+30))
				endif
			else
				if @isobject(%tab_name) then
					%tab_name=@getnextname(%tab_name)
					table {%tab_name}
					tabplace({%tab_name}, temp_out,  "A1", "A1", "H"+@str(place+30))
				else
					table {%tab_name}
					tabplace({%tab_name}, temp_out,  "A1", "A1", "H"+@str(place+30))
				endif
			endif
		endif
	endif
endif

{%tab_name}.setwidth(1) 22

if !show_out=1 then
	show {%tab_name}
endif

endsub

' ============================================================================================================================
' the logarithm likelihood function of the DCC(1,1) model

subroutine local loglike_dcc(vector logl_dcc, vector alf, matrix et_mat, scalar cnst_dcc, scalar radio_tdist_dcc, scalar cnst_dcc_t, matrix dcc_param_constraints, scalar grid_search)

!lb_a=dcc_param_constraints(1,1)
!ub_a=dcc_param_constraints(1,2)
!lb_b=dcc_param_constraints(2,1)
!ub_b=dcc_param_constraints(2,2)
!lb_v=dcc_param_constraints(3,1)
!ub_v=dcc_param_constraints(3,2)

!num_var=@columns(et_mat)
scalar num_obs=@rows(et_mat)

matrix cor_unc=@cor(et_mat)
matrix R=cor_unc
sym(!num_var) Q=@covp(et_mat)
!logl_dcc_1=0

	if radio_tdist_dcc=1 then
		if alf(3)<2 then
			if grid_search=0 then
				@uiprompt("Error: The likelihood function cannot be evaluated due to missing values or other errors."  + @chr(13) + "The degrees of freedom parameter=" +@str(alf(3)) + "<2")
				statusline Error: The likelihood function cannot be evaluated due to missing values or other errors
				stop
			else
				return
			endif
		endif
		if cnst_dcc_t=1 then		'restrictions on the degrees of freedom parameter for t-distribution
			!vdcc=(!lb_v+(!ub_v-!lb_v)*@exp(alf(3))/(1+@exp(alf(3))))
			!logl_dcc_1a=@gammalog((!vdcc+!num_var)/2)-@gammalog(!vdcc/2)-0.5*!num_var*@log((!vdcc-2)*@acos(-1))-0.5*@log(@det(R))
			!logl_dcc_1b=-0.5*(!vdcc+!num_var)*@log(1+(@rowextract(et_mat, 1)*@inverse(R)*@transpose(@rowextract(et_mat, 1)))/(!vdcc-2))		
		else			
			!logl_dcc_1a=@gammalog((alf(3)+!num_var)/2)-@gammalog(alf(3)/2)-0.5*!num_var*@log((alf(3)-2)*@acos(-1))-0.5*@log(@det(R))
			!logl_dcc_1b=-0.5*(alf(3)+!num_var)*@log(1+(@rowextract(et_mat, 1)*@inverse(R)*@transpose(@rowextract(et_mat, 1)))/(alf(3)-2))
		endif
		!logl_dcc_1=!logl_dcc_1a+!logl_dcc_1b
	else
'		!logl_dcc_1=-0.5*(@log(@det(R))+@rowextract(et_mat, 1)*@inverse(R)*@transpose(@rowextract(et_mat, 1))-@rowextract(et_mat,1)*@transpose(@rowextract(et_mat,1)))
		!logl_dcc_1=-0.5*(@log(@det(R))+@rowextract(et_mat, 1)*@inverse(R)*@transpose(@rowextract(et_mat, 1))-@inner(@rowextract(et_mat, 1), @rowextract(et_mat, 1)))
	endif	

	if @isna(!logl_dcc_1) then
		if grid_search=0 then
			@uiprompt("Error: The likelihood function cannot be evaluated due to missing values or other errors")
			statusline Error: The likelihood function cannot be evaluated due to missing values or other errors
			stop
		else
			return
		endif
	endif

logl_dcc(1)=!logl_dcc_1

if cnst_dcc=0 then														' no restrictions on the DCC parameters 

for !i4=2 to num_obs

sym Q=cor_unc*(1-alf(1)-alf(2))+alf(1)*(@transpose(@rowextract(et_mat,!i4-1))*@rowextract(et_mat,!i4-1))+alf(2)*Q
vector Qd=@getmaindiagonal(Q)

	for !i6=1 to !num_var
		if Qd(!i6)<0 then
			Qd(!i6)=0.01
		endif
	next

	if @nas(Qd)>0 then
		if grid_search=0 then
			@uiprompt("Error: The likelihood function cannot be evaluated due to missing values or other errors")
			statusline Error: The likelihood function cannot be evaluated due to missing values or other errors
			stop
		else
			return
		endif
	endif

sym Qs = @sqrt(@makediagonal(Qd))
sym Qi=@inverse(Qs)
sym Rn=@qform(Q, Qi)			'matrix Rn=@inverse(Qs)*Q*@inverse(Qs)
!logl_dcc_i=0

	if radio_tdist_dcc=1 then
			if cnst_dcc_t=1 then												'restrictions on the degrees of freedom parameter for t-distribution
				!vdcc=(!lb_v+(!ub_v-!lb_v)*@exp(alf(3))/(1+@exp(alf(3))))
				!logl_dcc_i1=@gammalog((!vdcc+!num_var)/2) -@gammalog(!vdcc/2) -0.5*!num_var*@log((!vdcc-2)*@acos(-1)) -0.5*@log(@det(Rn))
				!logl_dcc_i2= -0.5*(!vdcc+!num_var)*@log(1+(@rowextract(et_mat, !i4)*@inverse(Rn)*@transpose(@rowextract(et_mat, !i4)))/(!vdcc-2))
			else			
				!logl_dcc_i1=@gammalog((alf(3)+!num_var)/2) -@gammalog(alf(3)/2) -0.5*!num_var*@log((alf(3)-2)*@acos(-1)) -0.5*@log(@det(Rn))
				!logl_dcc_i2= -0.5*(alf(3)+!num_var)*@log(1+(@rowextract(et_mat, !i4)*@inverse(Rn)*@transpose(@rowextract(et_mat, !i4)))/(alf(3)-2))
			endif
		!logl_dcc_i=!logl_dcc_i1+!logl_dcc_i2
	else
'		!logl_dcc_i=-0.5*(@log(@det(Rn))+@rowextract(et_mat,!i4)*@inverse(Rn)*@transpose(@rowextract(et_mat,!i4))-@rowextract(et_mat,!i4)*@transpose(@rowextract(et_mat,!i4)))
		!logl_dcc_i=-0.5*(@log(@det(Rn))+@rowextract(et_mat,!i4)*@inverse(Rn)*@transpose(@rowextract(et_mat,!i4))-@inner(@rowextract(et_mat, !i4), @rowextract(et_mat, !i4)))
	endif

	if @isna(!logl_dcc_i) then
		if grid_search=0 then
			@uiprompt("Error: The likelihood function cannot be evaluated due to missing values or other errors")
			statusline Error: The likelihood function cannot be evaluated due to missing values or other errors
			stop
		else
			return
		endif
	endif

logl_dcc(!i4)=!logl_dcc_i

next

else

	if cnst_dcc=1 then														' restrictions on both of the DCC parameters, namely ALPHA and BETA

	for !i4=2 to num_obs
		!a1=(!lb_a+(!ub_a-!lb_a)*@exp(alf(1))/(1+@exp(alf(1))))
		!b1=(!lb_b+(!ub_b-!lb_b)*@exp(alf(2))/(1+@exp(alf(2))))
		sym Q=cor_unc*(1-!a1-!b1)+!a1*(@transpose(@rowextract(et_mat,!i4-1))*@rowextract(et_mat,!i4-1))+!b1*Q
		vector Qd=@getmaindiagonal(Q)

		for !i6=1 to !num_var
			if Qd(!i6)<0 then
				Qd(!i6)=0.01
			endif
		next

	if @nas(Qd)>0 then
		if grid_search=0 then
			@uiprompt("Error: The likelihood function cannot be evaluated due to missing values or other errors")
			statusline Error: The likelihood function cannot be evaluated due to missing values or other errors
			stop
		else
			return
		endif
	endif

	sym Qs = @sqrt(@makediagonal(Qd))
	sym Qi=@inverse(Qs)
	sym Rn=@qform(Q, Qi)				'matrix Rn=@inverse(Qs)*Q*@inverse(Qs)
	!logl_dcc_i=0

		if radio_tdist_dcc=1 then
			if cnst_dcc_t=1 then												'restrictions on the degrees of freedom parameter for t-distribution
				!vdcc=(!lb_v+(!ub_v-!lb_v)*@exp(alf(3))/(1+@exp(alf(3))))
				!logl_dcc_i1=@gammalog((!vdcc+!num_var)/2) - @gammalog(!vdcc/2) -0.5*!num_var*@log((!vdcc-2)*@acos(-1)) -0.5*@log(@det(Rn))
				!logl_dcc_i2= -0.5*(!vdcc+!num_var)*@log(1+(@rowextract(et_mat, !i4)*@inverse(Rn)*@transpose(@rowextract(et_mat, !i4)))/(!vdcc-2))
			else			
				!logl_dcc_i1=@gammalog((alf(3)+!num_var)/2) - @gammalog(alf(3)/2) - 0.5*!num_var*@log((alf(3)-2)*@acos(-1)) -0.5*@log(@det(Rn))
				!logl_dcc_i2= -0.5*(alf(3)+!num_var)*@log(1+(@rowextract(et_mat, !i4)*@inverse(Rn)*@transpose(@rowextract(et_mat, !i4)))/(alf(3)-2))
			endif
		
			!logl_dcc_i=!logl_dcc_i1+!logl_dcc_i2
		else
	'		!logl_dcc_i=-0.5*(@log(@det(Rn))+@rowextract(et_mat,!i4)*@inverse(Rn)*@transpose(@rowextract(et_mat,!i4))-@rowextract(et_mat,!i4)*@transpose(@rowextract(et_mat,!i4)))
			!logl_dcc_i=-0.5*(@log(@det(Rn))+@rowextract(et_mat,!i4)*@inverse(Rn)*@transpose(@rowextract(et_mat,!i4))-@inner(@rowextract(et_mat, !i4), @rowextract(et_mat, !i4)))
		endif

		if @isna(!logl_dcc_i) then
			if grid_search=0 then
				@uiprompt("Error: The likelihood function cannot be evaluated due to missing values or other errors")
				statusline Error: The likelihood function cannot be evaluated due to missing values or other errors
				stop
			else
				return
			endif
		endif	
	logl_dcc(!i4)=!logl_dcc_i
	next

	else

		if cnst_dcc=2 then												' restrictions only on the ALPHA parameter

		for !i4=2 to num_obs
		!a1=(!lb_a+(!ub_a-!lb_a)*@exp(alf(1))/(1+@exp(alf(1))))
		sym Q=cor_unc*(1-!a1-alf(2))+!a1*(@transpose(@rowextract(et_mat,!i4-1))*@rowextract(et_mat,!i4-1))+alf(2)*Q
		vector Qd=@getmaindiagonal(Q)

			for !i6=1 to !num_var
				if Qd(!i6)<0 then
					Qd(!i6)=0.01
				endif
			next

			if @nas(Qd)>0 then
				if grid_search=0 then
					@uiprompt("Error: The likelihood function cannot be evaluated due to missing values or other errors")
					statusline Error: The likelihood function cannot be evaluated due to missing values or other errors
					stop
				else
					return
				endif
			endif

		sym Qs = @sqrt(@makediagonal(Qd))
		sym Qi=@inverse(Qs)
		sym Rn=@qform(Q, Qi)				'matrix Rn=@inverse(Qs)*Q*@inverse(Qs)
		!logl_dcc_i=0

			if radio_tdist_dcc=1 then
					if cnst_dcc_t=1 then										'restrictions on the degrees of freedom parameter for t-distribution
						!vdcc=(!lb_v+(!ub_v-!lb_v)*@exp(alf(3))/(1+@exp(alf(3))))
						!logl_dcc_i1=@gammalog((!vdcc+!num_var)/2) -@gammalog(!vdcc/2) -0.5*!num_var*@log((!vdcc-2)*@acos(-1)) -0.5*@log(@det(Rn))
						!logl_dcc_i2= -0.5*(!vdcc+!num_var)*@log(1+(@rowextract(et_mat, !i4)*@inverse(Rn)*@transpose(@rowextract(et_mat, !i4)))/(!vdcc-2))
					else			
						!logl_dcc_i1=@gammalog((alf(3)+!num_var)/2) -@gammalog(alf(3)/2) -0.5*!num_var*@log((alf(3)-2)*@acos(-1)) -0.5*@log(@det(Rn))
						!logl_dcc_i2= -0.5*(alf(3)+!num_var)*@log(1+(@rowextract(et_mat, !i4)*@inverse(Rn)*@transpose(@rowextract(et_mat, !i4)))/(alf(3)-2))
					endif
				!logl_dcc_i=!logl_dcc_i1+!logl_dcc_i2
			else
		'		!logl_dcc_i=-0.5*(@log(@det(Rn))+@rowextract(et_mat,!i4)*@inverse(Rn)*@transpose(@rowextract(et_mat,!i4))-@rowextract(et_mat,!i4)*@transpose(@rowextract(et_mat,!i4)))
				!logl_dcc_i=-0.5*(@log(@det(Rn))+@rowextract(et_mat,!i4)*@inverse(Rn)*@transpose(@rowextract(et_mat,!i4))-@inner(@rowextract(et_mat, !i4), @rowextract(et_mat, !i4)))
			endif

			if @isna(!logl_dcc_i) then
				if grid_search=0 then
					@uiprompt("Error: The likelihood function cannot be evaluated due to missing values or other errors")
					statusline Error: The likelihood function cannot be evaluated due to missing values or other errors
					stop
				else
					return
				endif
			endif	

		logl_dcc(!i4)=!logl_dcc_i

		next
		
		else																			' restrictions only on the BETA parameter
		
		for !i4=2 to num_obs
		!b1=(!lb_b+(!ub_b-!lb_b)*@exp(alf(2))/(1+@exp(alf(2))))
		sym Q=cor_unc*(1-alf(1)-!b1)+alf(1)*(@transpose(@rowextract(et_mat,!i4-1))*@rowextract(et_mat,!i4-1))+!b1*Q
		vector Qd=@getmaindiagonal(Q)

			for !i6=1 to !num_var
				if Qd(!i6)<0 then
					Qd(!i6)=0.01
				endif
			next

			if @nas(Qd)>0 then
				if grid_search=0 then
					@uiprompt("Error: The likelihood function cannot be evaluated due to missing values or other errors")
					statusline Error: The likelihood function cannot be evaluated due to missing values or other errors
					stop
				else
					return
				endif
			endif

		sym Qs = @sqrt(@makediagonal(Qd))
		sym Qi=@inverse(Qs)
		sym Rn=@qform(Q, Qi)				'matrix Rn=@inverse(Qs)*Q*@inverse(Qs)
		!dcc_on=0

			if radio_tdist_dcc=1 then
					if cnst_dcc_t=1 then		'restrictions on the degrees of freedom parameter for t-distribution
						!vdcc=(!lb_v+(!ub_v-!lb_v)*@exp(alf(3))/(1+@exp(alf(3))))
						!logl_dcc_i1=@gammalog((!vdcc+!num_var)/2) -@gammalog(!vdcc/2) -0.5*!num_var*@log((!vdcc-2)*@acos(-1)) -0.5*@log(@det(Rn))
						!logl_dcc_i2= -0.5*(!vdcc+!num_var)*@log(1+(@rowextract(et_mat, !i4)*@inverse(Rn)*@transpose(@rowextract(et_mat, !i4)))/(!vdcc-2))
					else			
						!logl_dcc_i1=@gammalog((alf(3)+!num_var)/2) -@gammalog(alf(3)/2) -0.5*!num_var*@log((alf(3)-2)*@acos(-1)) -0.5*@log(@det(Rn))
						!logl_dcc_i2= -0.5*(alf(3)+!num_var)*@log(1+(@rowextract(et_mat, !i4)*@inverse(Rn)*@transpose(@rowextract(et_mat, !i4)))/(alf(3)-2))
					endif
				!logl_dcc_i=!logl_dcc_i1+!logl_dcc_i2
			else
'				!logl_dcc_i=-0.5*(@log(@det(Rn))+@rowextract(et_mat,!i4)*@inverse(Rn)*@transpose(@rowextract(et_mat,!i4))-@rowextract(et_mat,!i4)*@transpose(@rowextract(et_mat,!i4)))
				!logl_dcc_i=-0.5*(@log(@det(Rn))+@rowextract(et_mat,!i4)*@inverse(Rn)*@transpose(@rowextract(et_mat,!i4))-@inner(@rowextract(et_mat, !i4), @rowextract(et_mat, !i4)))
			endif

			if @isna(!logl_dcc_i) then
				if grid_search=0 then
					@uiprompt("Error: The likelihood function cannot be evaluated due to missing values or other errors")
					statusline Error: The likelihood function cannot be evaluated due to missing values or other errors
					stop
				else
					return
				endif
			endif
		logl_dcc(!i4)=!logl_dcc_i
		next
		endif
	endif
endif

endsub


