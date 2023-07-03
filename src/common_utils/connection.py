from typing import Any
from urllib.parse import quote_plus

import cx_Oracle as cxo
import pandas as pd
from omegaconf import DictConfig
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import PoolProxiedConnection
from sshtunnel import SSHTunnelForwarder
from misc import check


def execute_query(statement: str,
                  config: DictConfig) -> pd.DataFrame:
    """Execute query

    Arguments
    ---------
    `statement: str`
        Statement.

    `connection: PoolProxiedConnection | Engine`
        Connection to BD.

    Returns
    -------
    `data: pd.DataFrame`
        Result of query.
    """
    print(statement, flush=True)
    data: pd.DataFrame = pd.read_sql(
        statement,
        con=_connect(config=config, type='select')
    )
    if 'ssh' in config.keys():
        server.close()

    return data


def execute_statements(statements: str | list[str],
                       config: DictConfig) -> None:
    """Execute statement

    Execute insert or update statement.

    Arguments
    ---------
    `statements: str | list[str]`
        Statement or list of statements.

    `config: DictConfig`
        Connection configuration.
    """
    connection = _connect(config=config, type='update')

    if type(statements) == str:
        statements = [statements]

    cursor = connection.cursor()
    for statement in statements:
        cursor.execute(statement)
    connection.commit()
    connection.close()


def _connect(
    config: DictConfig,
    type: str = 'select'
) -> PoolProxiedConnection | cxo.Connection:
    """Connect

    Create a connection engine. The type of connector depends on the use of
    connection, to assure compatibility with `pd.read_sql` and `pd.to_sql`.

    Parameters
    ----------
    `config: DictConfig`
        Object with configurations. It must have the keys `drivername`,
        `user`, `password`, `host`, `port`. Is must also have either
        `service` or `database` keys.

    `type: str`
        Type of connection. It must be `select`, `insert` or `update`. If it
        is `select`, an `sqlalchemy.pool.PoolProxiedConnection is returned. In
        other cases it returns an `sqlalchemy.engine.Engine`.

    Returns
    -------
    `connection: PoolProxiedConnection | Engine`
        The connection.

    Raises
    ------
    ValueError
        If `type` is not one of `select`, `insert` or `update` or if there are
        missing keys in `config`.
    """
    # Check parameter validity
    check(config=config)
    if type not in ['select', 'insert', 'update']:
        msg = (
            f'Accepted values for parameter `type` are `"select"`, `"insert"`'
            f' and `"update"`. However, value {type} was passed.'
        )
        raise ValueError(msg)

    # Make connection URL
    drivername: str = config.drivername
    user: str = config.user
    password: str = quote_plus(config.password)
    host: str = config.host
    port: str = config.port
    if 'database' in config:
        database = config.database
    else:
        database = ''
    if 'service_name' in config:
        service_name = config.service_name
    else:
        service_name = ''

    if 'ssh' in config.keys():
        # If SSH is to be used, port is forwarded. Otherwise, nothing is done
        global server
        server = SSHTunnelForwarder(
            (config.ssh.host),
            ssh_username=config.ssh.username,
            ssh_password=config.ssh.password,
            remote_bind_address=('127.0.0.1', config.port)
        )
        server.start()
        port = server.local_bind_port
    else:
        port = config.port

    if type == 'select' or config.drivername == 'mysql':
        url: str = f"{drivername}://{user}:{password}@{host}:{port}/{database}"
        if 'service_name' in config:
            url = url + f'?service_name={service_name}'
        connection: PoolProxiedConnection = \
            create_engine(url).raw_connection()
        return connection
    else:
        if 'service_name' in config.keys():
            dsn = cxo.makedsn(host, port, service_name=service_name)
        else:
            dsn = cxo.makedsn(host, port, database)
        connection_cxo: cxo.Connection = cxo.connect(user=user,
                                                     password=password,
                                                     dsn=dsn)
        return connection_cxo
