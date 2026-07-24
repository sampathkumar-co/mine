"use client";

import {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { getAccountContext } from "@/lib/account";
import {
  getAccessToken,
  getSession,
  login as apiLogin,
  logout as apiLogout,
  register as apiRegister,
  setSessionTokens,
} from "@/lib/api";
import type { AuthSession } from "@/lib/types";

interface AuthContextValue {
  session: AuthSession | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (input: {
    email: string;
    password: string;
    displayName: string;
    workspaceName: string;
  }) => Promise<void>;
  logout: () => Promise<void>;
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

async function hydrateSession(session: AuthSession): Promise<AuthSession> {
  const account = await getAccountContext();
  return { ...session, user: account.user, workspaces: account.workspaces };
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<AuthSession | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      setSession(await hydrateSession(await getSession()));
    } catch {
      setSessionTokens(null);
      setSession(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const changed = () => {
      if (!getAccessToken()) setSession(null);
    };
    window.addEventListener("director-auth-changed", changed);
    return () => window.removeEventListener("director-auth-changed", changed);
  }, [refresh]);

  const value = useMemo<AuthContextValue>(
    () => ({
      session,
      loading,
      login: async (email, password) => {
        setSession(await hydrateSession(await apiLogin(email, password)));
      },
      register: async (input) => {
        setSession(await hydrateSession(await apiRegister(input)));
      },
      logout: async () => {
        await apiLogout();
        setSession(null);
      },
      refresh,
    }),
    [loading, refresh, session],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used inside AuthProvider");
  return value;
}
