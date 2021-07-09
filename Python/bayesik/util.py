




def report_sim_iteration(q_true, q_estimated, absolute_q=False, labels=('LS-IK','B-IK')):
	nq          = q_true.size
	fmt         = '%.5f ' * nq
	print()
	if absolute_q:
		print( 'True:'.ljust(15) + fmt %tuple(q_true) )
		for q,s in zip(q_estimated, labels):
			print( f'{s}:'.ljust(15) + fmt %tuple(q) )
		print('---------------')
	### report errors:
	for q,s in zip(q_estimated, labels):
		e = q - q_true
		print( f'{s}:'.ljust(15) + fmt %tuple(e) )


def report_sim_summary(Q_true, Q_estimated, i, T, labels=('LS-IK','B-IK')):
	nq           = Q_true.shape[1]
	fmt          = '%.5f ' * nq
	ii           = i+1
	print('\n\n\n')
	print('----- SUMMARY (to iteration %d)----' %i)
	print('------------ RMSE -----------------')
	for Q,s in zip(Q_estimated, labels):
		E    = Q[:ii] - Q_true[:ii]
		RMSE = (E**2).mean(axis=0)**0.5
		print( f'{s}:'.ljust(15) + fmt %tuple(RMSE) )
	print('------------ Average durations ----')
	for t,s in zip(T, labels):
		dt   = t[:ii].mean()
		print( f'{s} (s):'.ljust(20) + '%.5f'%dt )
	
