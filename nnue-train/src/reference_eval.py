"""Standalone reference implementation of the king-relative NNUE evaluation.

numpy only -- no torch, no python-chess in the core. This is the engine-side contract in
executable form: if fastnnue.py agrees with this on spec/test_vectors.json, the net will
evaluate correctly.

Core API:
    net = KingRelativeNNUE.load_float("nnue_v1.npz")     # or load_quantised(".nnue")
    cp  = net.evaluate(pieces, stm)

where `pieces` is {square: (piece_type, colour)} with square a1=0..h8=63,
piece_type P,N,B,R,Q,K = 0..5, colour 0=white 1=black; stm 0=white 1=black.
Returns centipawns from the side to move's point of view.
"""
import numpy as np

NUM_FEATURES = 24576
EVAL_SCALE = 400.0
QH = 1024          # hidden weight scale; QA is read from the file header
WHITE, BLACK = 0, 1
KING = 5


def feature_indices(pieces, perspective):
    """Active feature indices for one perspective. See spec/king_relative_v1.md."""
    king_sq = next(sq for sq, (pt, c) in pieces.items()
                   if pt == KING and c == perspective)
    flip = 56 if perspective == BLACK else 0
    mirror = ((king_sq ^ flip) & 7) >= 4

    def transform(sq):
        s = sq ^ flip
        return s ^ 7 if mirror else s

    tksq = transform(king_sq)
    king_bucket = (tksq >> 3) * 4 + (tksq & 7)
    base = king_bucket * 768
    out = []
    for sq, (pt, colour) in pieces.items():
        rel = 0 if colour == perspective else 1
        out.append(base + (rel * 6 + pt) * 64 + transform(sq))
    return out


class KingRelativeNNUE:
    def __init__(self, ft_w, ft_b, l1, l2, l3, quantised=False, qa=None):
        self.ft_w, self.ft_b = ft_w, ft_b
        self.l1, self.l2, self.l3 = l1, l2, l3
        self.quantised = quantised
        self.qa = qa

    @classmethod
    def load_float(cls, path):
        z = np.load(path)
        return cls(z["ft.weight"], z["ft.bias"],
                   (z["lin1.weight"], z["lin1.bias"]),
                   (z["lin2.weight"], z["lin2.bias"]),
                   (z["lin3.weight"], z["lin3.bias"]), quantised=False)

    @classmethod
    def load_quantised(cls, path):
        with open(path, "rb") as f:
            assert f.read(4) == b"2PQN", "bad magic"
            _ver, nfeat, l1w, qa, qh = np.frombuffer(f.read(20), dtype="<i4")
            assert nfeat == NUM_FEATURES, f"feature count {nfeat}"
            assert qh == QH, f"hidden scale {qh} != {QH}"
            ft_w = np.frombuffer(f.read(nfeat * l1w * 2), dtype="<i2").reshape(nfeat, l1w)
            ft_b = np.frombuffer(f.read(l1w * 2), dtype="<i2")
            def layer(o, i):
                w = np.frombuffer(f.read(o * i * 2), dtype="<i2").reshape(o, i)
                b = np.frombuffer(f.read(o * 4), dtype="<i4")
                return w, b
            a = layer(32, 2 * l1w)
            b = layer(32, 32)
            c = layer(1, 32)
        return cls(ft_w, ft_b, a, b, c, quantised=True, qa=int(qa))

    # ---- accumulator ----
    def accumulate(self, pieces, perspective):
        idx = feature_indices(pieces, perspective)
        if self.quantised:
            return self.ft_b.astype(np.int32) + self.ft_w[idx].sum(axis=0, dtype=np.int32)
        return self.ft_b.astype(np.float32) + self.ft_w[idx].sum(axis=0, dtype=np.float32)

    def evaluate(self, pieces, stm):
        a_stm = self.accumulate(pieces, stm)
        a_nstm = self.accumulate(pieces, 1 - stm)
        x = np.concatenate([a_stm, a_nstm])
        if self.quantised:
            qa = self.qa
            x = np.clip(x, 0, qa)
            for w, b in (self.l1, self.l2):
                r = w.astype(np.int32) @ x + b
                x = np.clip((r + QH // 2) // QH, 0, qa)     # round, do not truncate
            w, b = self.l3
            out = (int((w.astype(np.int32) @ x)[0]) + int(b[0])) / (qa * QH)
        else:
            x = np.clip(x, 0.0, 1.0)
            for w, b in (self.l1, self.l2):
                x = np.clip(w @ x + b, 0.0, 1.0)
            w, b = self.l3
            out = float((w @ x + b)[0])
        return out * EVAL_SCALE

    # ---- incremental update ----
    def king_context(self, king_sq, perspective):
        """Precompute the per-perspective transform once, not once per piece.

        The engine should cache this alongside the accumulator and only recompute it when
        that perspective's own king moves.
        """
        flip = 56 if perspective == BLACK else 0
        mirror = ((king_sq ^ flip) & 7) >= 4
        tksq = (king_sq ^ flip)
        if mirror:
            tksq ^= 7
        base = ((tksq >> 3) * 4 + (tksq & 7)) * 768
        return base, flip, mirror

    @staticmethod
    def index_in(ctx, sq, piece_type, colour, perspective):
        base, flip, mirror = ctx
        t = sq ^ flip
        if mirror:
            t ^= 7
        rel = 0 if colour == perspective else 1
        return base + (rel * 6 + piece_type) * 64 + t

    def update_accumulator(self, acc, perspective, king_sq, removed, added,
                           own_king_moved=False, pieces=None):
        """Apply one move to a single perspective's accumulator.

        `removed` / `added` are lists of (square, piece_type, colour) -- note that a
        capture contributes the captured piece to `removed`, castling moves two pieces,
        and a promotion removes a pawn and adds the promoted piece.

        If this perspective's OWN king moved, every feature for it changes (the king
        square is the index anchor, and crossing the d/e file boundary also flips the
        mirror), so the accumulator must be rebuilt: pass own_king_moved=True and the
        full `pieces` map. The other perspective is unaffected and stays incremental.
        """
        if own_king_moved:
            return self.accumulate(pieces, perspective)
        ctx = self.king_context(king_sq, perspective)
        acc = acc.copy()
        for sq, pt, colour in removed:
            acc -= self.ft_w[self.index_in(ctx, sq, pt, colour, perspective)]
        for sq, pt, colour in added:
            acc += self.ft_w[self.index_in(ctx, sq, pt, colour, perspective)]
        return acc


def pieces_from_fen(fen):
    """Convenience adapter: FEN -> (pieces dict, stm). No python-chess needed."""
    board_field, stm_field = fen.split()[0], fen.split()[1]
    code = {c: i for i, c in enumerate("PNBRQK")}
    pieces = {}
    rank, file = 7, 0
    for ch in board_field:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            pt = code[ch.upper()]
            colour = WHITE if ch.isupper() else BLACK
            pieces[rank * 8 + file] = (pt, colour)
            file += 1
    return pieces, (BLACK if stm_field == "b" else WHITE)
