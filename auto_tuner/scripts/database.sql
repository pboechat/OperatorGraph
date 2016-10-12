create table partition(
	pk			integer		primary key asc, 
	idx			integer		not null,
	uid			text		not null, 
	num_edges	integer		not null, 
	instr		integer		not null, 
	opt			integer		not null, 
	exec_time	real		not null, 
	stdev		real		not null, 
	num_axioms  integer		not null
);

create unique index partition_idx_instr_opt on partition(idx, instr, opt, num_axioms);
create unique index partition_uid_instr_opt on partition(uid, instr, opt, num_axioms);

create table edge(
	pk				integer		primary key asc, 
	partition_pk	integer		not null, 
	idx				integer		not null,
	freq			integer		not null, 
	exec_time		real		not null,
	stdev			real		not null,
	type			integer		not null,
	subgraph_pk		integer		not null,
	foreign key(partition_pk) references partition(pk),
	foreign key(subgraph_pk) references subgraph(pk)
);

create unique index partition_edge_idx on edge(partition_pk, idx);

create table subgraph(
	pk				integer		primary key asc, 
	partition_pk	integer		not null, 
	idx				integer		not null,
	freq			integer		not null, 
	exec_time		real		not null,
	stdev			real		not null,
	foreign key(partition_pk) references partition(pk)
);

create unique index partition_subgraph_idx on subgraph(partition_pk, idx);
