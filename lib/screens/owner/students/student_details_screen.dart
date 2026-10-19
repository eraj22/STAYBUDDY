import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../api.dart';
import '../../../theme.dart';
import '../../../widgets/responsive_shell.dart';

class StudentDetailsScreen extends StatelessWidget {
  final Map<String, dynamic> student;
  final String hostelName;

  const StudentDetailsScreen({
    super.key,
    required this.student,
    required this.hostelName,
  });

  @override
  Widget build(BuildContext context) {
    final bool isActive = student['status'] == 'Active';
    final bool feeDue = student['feeStatus'] == 'Due';
    final String feeStatus = student['feeStatus']?.toString() ?? 'No invoice';
    final bool parentLinked = student['parentLinked'] == true;

    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: AppTheme.bg,
        body: CustomScrollView(
          physics: const BouncingScrollPhysics(),
          slivers: [
            SliverToBoxAdapter(
              child: _buildHeader(
                context,
                isActive: isActive,
              ),
            ),

            SliverPadding(
              padding: const EdgeInsets.fromLTRB(18, 18, 18, 0),
              sliver: SliverToBoxAdapter(
                child: _buildQuickInfo(
                  feeStatus: feeStatus,
                  parentLinked: parentLinked,
                ),
              ),
            ),

            SliverPadding(
              padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
              sliver: SliverToBoxAdapter(
                child: _sectionTitle(
                  title: 'Student Information',
                  subtitle: 'Personal and contact details',
                ),
              ),
            ),

            SliverPadding(
              padding: const EdgeInsets.symmetric(horizontal: 18),
              sliver: SliverToBoxAdapter(
                child: _buildInformationCard(),
              ),
            ),

            SliverPadding(
              padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
              sliver: SliverToBoxAdapter(
                child: _sectionTitle(
                  title: 'Accommodation',
                  subtitle: 'Current room and bed assignment',
                ),
              ),
            ),

            SliverPadding(
              padding: const EdgeInsets.symmetric(horizontal: 18),
              sliver: SliverToBoxAdapter(
                child: _buildAccommodationCard(
                  context,
                ),
              ),
            ),

            SliverPadding(
              padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
              sliver: SliverToBoxAdapter(
                child: _sectionTitle(
                  title: 'Guardian & Parent',
                  subtitle: 'Linked family information',
                ),
              ),
            ),

            SliverPadding(
              padding: const EdgeInsets.symmetric(horizontal: 18),
              sliver: SliverToBoxAdapter(
                child: _buildGuardianCard(
                  parentLinked: parentLinked,
                ),
              ),
            ),

            SliverPadding(
              padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
              sliver: SliverToBoxAdapter(
                child: _sectionTitle(
                  title: 'Fee Status',
                  subtitle: 'Monthly payment overview',
                ),
              ),
            ),

            SliverPadding(
              padding: const EdgeInsets.symmetric(horizontal: 18),
              sliver: SliverToBoxAdapter(
                child: _buildFeeCard(
                  feeStatus: feeStatus,
                ),
              ),
            ),

            const SliverToBoxAdapter(child: SizedBox(height: 40)),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(
    BuildContext context, {
    required bool isActive,
  }) {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Color(0xFF073F42),
            Color(0xFF0A696D),
            Color(0xFF0A797B),
          ],
        ),
        borderRadius: BorderRadius.vertical(
          bottom: Radius.circular(34),
        ),
      ),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(
            18,
            14,
            18,
            26,
          ),
          child: Column(
            children: [
              Row(
                children: [
                  _circleButton(
                    icon: Icons.arrow_back_rounded,
                    onTap: () => Navigator.pop(context),
                  ),

                  const Spacer(),

                  _circleButton(
                    icon: Icons.more_horiz_rounded,
                    onTap: () {
                      _showMoreMenu(context);
                    },
                  ),
                ],
              ),

              const SizedBox(height: 24),

              _buildAvatar(),

              const SizedBox(height: 14),

              Text(
                student['name']?.toString() ?? 'Student',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                textAlign: TextAlign.center,
                style: GoogleFonts.playfairDisplay(
                  fontSize: 27,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),

              const SizedBox(height: 6),

              Text(
                student['email']?.toString() ?? '',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                textAlign: TextAlign.center,
                style: GoogleFonts.dmSans(
                  fontSize: 11.5,
                  color: Colors.white.withOpacity(0.65),
                ),
              ),

              const SizedBox(height: 12),

              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  _headerChip(
                    icon: isActive
                        ? Icons.verified_rounded
                        : Icons.pause_circle_rounded,
                    text: isActive ? 'Active' : 'Inactive',
                    color: isActive
                        ? AppTheme.tealMint
                        : Colors.orange.shade200,
                  ),

                  const SizedBox(width: 8),

                  _headerChip(
                    icon: Icons.apartment_rounded,
                    text: hostelName,
                    color: Colors.white,
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildAvatar() {
    final name = student['name']?.toString().trim() ?? 'S';
    final parts = name.split(' ');

    String initials =
        parts.isNotEmpty && parts.first.isNotEmpty
            ? parts.first[0].toUpperCase()
            : 'S';

    if (parts.length > 1 && parts.last.isNotEmpty) {
      initials += parts.last[0].toUpperCase();
    }

    return Stack(
      clipBehavior: Clip.none,
      children: [
        Container(
          width: 92,
          height: 92,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: Colors.white.withOpacity(0.12),
            border: Border.all(
              color: Colors.white.withOpacity(0.25),
              width: 2,
            ),
          ),
          child: Center(
            child: Text(
              initials,
              style: GoogleFonts.dmSans(
                fontSize: 28,
                fontWeight: FontWeight.w800,
                color: Colors.white,
              ),
            ),
          ),
        ),

        Positioned(
          bottom: 2,
          right: 2,
          child: Container(
            width: 26,
            height: 26,
            decoration: BoxDecoration(
              color: AppTheme.tealMint,
              shape: BoxShape.circle,
              border: Border.all(
                color: const Color(0xFF0A696D),
                width: 2,
              ),
            ),
            child: const Icon(
              Icons.check_rounded,
              size: 15,
              color: Color(0xFF073F42),
            ),
          ),
        ),
      ],
    );
  }

  Widget _circleButton({
    required IconData icon,
    required VoidCallback onTap,
  }) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        customBorder: const CircleBorder(),
        child: Container(
          width: 42,
          height: 42,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.13),
            shape: BoxShape.circle,
            border: Border.all(
              color: Colors.white.withOpacity(0.18),
            ),
          ),
          child: Icon(
            icon,
            color: Colors.white,
            size: 20,
          ),
        ),
      ),
    );
  }

  Widget _headerChip({
    required IconData icon,
    required String text,
    required Color color,
  }) {
    return Container(
      constraints: const BoxConstraints(
        maxWidth: 160,
      ),
      padding: const EdgeInsets.symmetric(
        horizontal: 10,
        vertical: 6,
      ),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.10),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            size: 13,
            color: color,
          ),
          const SizedBox(width: 5),
          Flexible(
            child: Text(
              text,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.dmSans(
                fontSize: 10,
                fontWeight: FontWeight.w600,
                color: color,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildQuickInfo({
    required String feeStatus,
    required bool parentLinked,
  }) {
    final room = student['room']?.toString() ?? 'Unassigned';
    final bed = student['bed']?.toString() ?? '-';

    return LayoutBuilder(
      builder: (context, constraints) {
        final cardWidth =
            (constraints.maxWidth - 12) / 2;

        return Wrap(
          spacing: 12,
          runSpacing: 12,
          children: [
            SizedBox(
              width: cardWidth,
              child: _quickInfoCard(
                icon: Icons.bed_rounded,
                label: 'Room & Bed',
                value: room == 'Unassigned'
                    ? 'Unassigned'
                    : '$room · $bed',
                color: const Color(0xFF3478B7),
                background:
                    const Color(0xFFEAF3FB),
              ),
            ),

            SizedBox(
              width: cardWidth,
              child: _quickInfoCard(
                icon: Icons.receipt_long_rounded,
                label: 'Fee Status',
                value: feeStatus == 'Paid'
                    ? 'Paid'
                    : feeStatus == 'Due'
                    ? 'Payment Due'
                    : 'No invoice',
                color: feeStatus == 'Paid'
                    ? const Color(0xFF3D7B53)
                    : feeStatus == 'Due'
                    ? const Color(0xFFC07421)
                    : AppTheme.textMuted,
                background: feeStatus == 'Paid'
                    ? const Color(0xFFEDF6EF)
                    : feeStatus == 'Due'
                    ? const Color(0xFFFFF4E6)
                    : AppTheme.bgSecondary,
              ),
            ),

            SizedBox(
              width: cardWidth,
              child: _quickInfoCard(
                icon: Icons.family_restroom_rounded,
                label: 'Parent',
                value: parentLinked
                    ? 'Linked'
                    : 'Not linked',
                color: parentLinked
                    ? AppTheme.teal
                    : AppTheme.textMuted,
                background: parentLinked
                    ? AppTheme.teal.withOpacity(0.07)
                    : AppTheme.bgSecondary,
              ),
            ),

            SizedBox(
              width: cardWidth,
              child: _quickInfoCard(
                icon: Icons.calendar_month_rounded,
                label: 'Joined',
                value:
                    student['joinDate']?.toString() ??
                    'Not available',
                color: const Color(0xFF7D55B4),
                background:
                    const Color(0xFFF2ECFB),
              ),
            ),
          ],
        );
      },
    );
  }

  Widget _quickInfoCard({
    required IconData icon,
    required String label,
    required String value,
    required Color color,
    required Color background,
  }) {
    return Container(
      padding: const EdgeInsets.all(15),
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: AppTheme.border,
          width: 0.6,
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              color: background,
              borderRadius: BorderRadius.circular(13),
            ),
            child: Icon(
              icon,
              color: color,
              size: 20,
            ),
          ),

          const SizedBox(width: 11),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  value,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    fontSize: 12.5,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.textPrimary,
                  ),
                ),

                const SizedBox(height: 2),

                Text(
                  label,
                  style: GoogleFonts.dmSans(
                    fontSize: 9.5,
                    color: AppTheme.textMuted,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _sectionTitle({
    required String title,
    required String subtitle,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: GoogleFonts.playfairDisplay(
            fontSize: 19,
            fontWeight: FontWeight.w700,
            color: AppTheme.textPrimary,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          subtitle,
          style: GoogleFonts.dmSans(
            fontSize: 10.5,
            color: AppTheme.textMuted,
          ),
        ),
      ],
    );
  }

  Widget _buildInformationCard() {
    return _sectionCard(
      children: [
        _infoRow(
          icon: Icons.person_outline_rounded,
          label: 'Full Name',
          value: student['name']?.toString() ?? '-',
        ),
        _divider(),
        _infoRow(
          icon: Icons.badge_outlined,
          label: 'CNIC / B-Form',
          value: student['cnic']?.toString().isNotEmpty == true ? student['cnic'].toString() : '-',
        ),
        _divider(),
        _infoRow(
          icon: Icons.school_outlined,
          label: 'University / Department',
          value: [student['university'], student['department']].where((value) => value?.toString().isNotEmpty == true).join(' · ').isNotEmpty ? [student['university'], student['department']].where((value) => value?.toString().isNotEmpty == true).join(' · ') : '-',
        ),
        _divider(),
        _infoRow(
          icon: Icons.email_outlined,
          label: 'Email',
          value: student['email']?.toString() ?? '-',
        ),
        _divider(),
        _infoRow(
          icon: Icons.phone_outlined,
          label: 'Phone',
          value: student['phone']?.toString() ?? '-',
        ),
        _divider(),
        _infoRow(
          icon: Icons.location_city_outlined,
          label: 'City',
          value: student['city']?.toString() ?? '-',
        ),
        _divider(),
        _infoRow(
          icon: Icons.calendar_month_outlined,
          label: 'Admission Date',
          value: student['joinDate']?.toString() ?? '-',
        ),
      ],
    );
  }

  Widget _buildAccommodationCard(
    BuildContext context,
  ) {
    final room =
        student['room']?.toString() ?? 'Unassigned';
    final bed = student['bed']?.toString() ?? '-';
    final isAssigned = room != 'Unassigned';

    return Container(
      padding: const EdgeInsets.all(17),
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: AppTheme.border,
        ),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Container(
                width: 46,
                height: 46,
                decoration: BoxDecoration(
                  color: AppTheme.teal.withOpacity(0.07),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(
                  Icons.apartment_rounded,
                  color: AppTheme.teal,
                  size: 22,
                ),
              ),

              const SizedBox(width: 12),

              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      isAssigned
                          ? 'Room $room'
                          : 'No room assigned',
                      style: GoogleFonts.dmSans(
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                        color: AppTheme.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      isAssigned
                          ? 'Bed $bed · $hostelName'
                          : 'Assign a room and bed to this student',
                      style: GoogleFonts.dmSans(
                        fontSize: 10.5,
                        color: AppTheme.textMuted,
                      ),
                    ),
                  ],
                ),
              ),

              if (isAssigned)
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 9,
                    vertical: 5,
                  ),
                  decoration: BoxDecoration(
                    color:
                        const Color(0xFFEDF6EF),
                    borderRadius:
                        BorderRadius.circular(20),
                  ),
                  child: Text(
                    'Assigned',
                    style: GoogleFonts.dmSans(
                      fontSize: 9,
                      fontWeight: FontWeight.w700,
                      color:
                          const Color(0xFF3D7B53),
                    ),
                  ),
                ),
            ],
          ),

          const SizedBox(height: 17),

          SizedBox(
            width: double.infinity,
            height: 48,
            child: OutlinedButton.icon(
              onPressed: () {
                _temporaryMessage(
                  context,
                  isAssigned
                      ? 'Change Room screen will be connected next.'
                      : 'Room Assignment screen will be connected next.',
                );
              },
              icon: Icon(
                isAssigned
                    ? Icons.swap_horiz_rounded
                    : Icons.add_home_work_outlined,
                size: 18,
              ),
              label: Text(
                isAssigned
                    ? 'Change Room / Bed'
                    : 'Assign Room / Bed',
                style: GoogleFonts.dmSans(
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                ),
              ),
              style: OutlinedButton.styleFrom(
                foregroundColor: AppTheme.teal,
                side: BorderSide(
                  color: AppTheme.teal.withOpacity(0.45),
                ),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(15),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildGuardianCard({
    required bool parentLinked,
  }) {
    return _sectionCard(
      children: [
        _infoRow(
          icon: Icons.family_restroom_rounded,
          label: 'Guardian',
          value:
              student['guardian']?.toString() ??
              'Not provided',
        ),
        _divider(),
        _infoRow(
          icon: Icons.phone_outlined,
          label: 'Guardian contact',
          value: [student['parentRelation'], student['parentPhone'], student['parentEmail']].where((value) => value?.toString().isNotEmpty == true).join(' · ').isNotEmpty ? [student['parentRelation'], student['parentPhone'], student['parentEmail']].where((value) => value?.toString().isNotEmpty == true).join(' · ') : 'Not provided',
        ),
        _divider(),
        Row(
          children: [
            Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(
                color: parentLinked
                    ? AppTheme.teal.withOpacity(0.07)
                    : AppTheme.bgSecondary,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(
                parentLinked
                    ? Icons.link_rounded
                    : Icons.link_off_rounded,
                size: 18,
                color: parentLinked
                    ? AppTheme.teal
                    : AppTheme.textMuted,
              ),
            ),

            const SizedBox(width: 12),

            Expanded(
              child: Column(
                crossAxisAlignment:
                    CrossAxisAlignment.start,
                children: [
                  Text(
                    parentLinked
                        ? 'Parent account linked'
                        : 'Parent not linked',
                    style: GoogleFonts.dmSans(
                      fontSize: 12.5,
                      fontWeight: FontWeight.w600,
                      color: AppTheme.textPrimary,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    parentLinked
                        ? 'Parent can access student information'
                        : 'Generate invitation for parent access',
                    style: GoogleFonts.dmSans(
                      fontSize: 9.5,
                      color: AppTheme.textMuted,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildFeeCard({
    required String feeStatus,
  }) {
    final feeDue = feeStatus == 'Due';
    final paid = feeStatus == 'Paid';
    final color = feeDue
        ? const Color(0xFFC07421)
        : paid
        ? const Color(0xFF3D7B53)
        : AppTheme.textMuted;

    return Container(
      padding: const EdgeInsets.all(17),
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: AppTheme.border,
        ),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Container(
                width: 46,
                height: 46,
                decoration: BoxDecoration(
                  color: color.withOpacity(0.09),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(
                  feeDue ? Icons.schedule_rounded : paid ? Icons.check_circle_rounded : Icons.receipt_long_outlined,
                  size: 22,
                  color: color,
                ),
              ),

              const SizedBox(width: 12),

              Expanded(
                child: Column(
                  crossAxisAlignment:
                      CrossAxisAlignment.start,
                  children: [
                    Text(
                      feeDue ? 'Payment Pending' : paid ? 'All dues cleared' : 'No fee invoice yet',
                      style: GoogleFonts.dmSans(
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                        color: AppTheme.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      feeDue ? 'Current month hostel fee is pending' : paid ? 'No pending hostel fee' : 'A fee invoice has not been generated for this student.',
                      style: GoogleFonts.dmSans(
                        fontSize: 10.5,
                        color: AppTheme.textMuted,
                      ),
                    ),
                  ],
                ),
              ),

              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 9,
                  vertical: 5,
                ),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.09),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  feeDue ? 'Due' : 'Paid',
                  style: GoogleFonts.dmSans(
                    fontSize: 9,
                    fontWeight: FontWeight.w700,
                    color: color,
                  ),
                ),
              ),
            ],
          ),

          const SizedBox(height: 17),

          Row(
            children: [
              Expanded(
                child: _feeMiniCard(
                  label: 'Monthly Fee',
                  value: 'PKR 15,000',
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _feeMiniCard(
                  label: 'Due Date',
                  value: '10 Aug',
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _feeMiniCard({
    required String label,
    required String value,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: 12,
        vertical: 12,
      ),
      decoration: BoxDecoration(
        color: AppTheme.bgSecondary,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          Text(
            value,
            style: GoogleFonts.dmSans(
              fontSize: 12.5,
              fontWeight: FontWeight.w700,
              color: AppTheme.textPrimary,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            label,
            style: GoogleFonts.dmSans(
              fontSize: 9.5,
              color: AppTheme.textMuted,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAccountCard(
    BuildContext context, {
    required bool isActive,
  }) {
    return _sectionCard(
      children: [
        _actionRow(
          icon: Icons.password_rounded,
          title: 'Generate Credentials',
          subtitle:
              'Create student login username and password',
          color: AppTheme.teal,
          onTap: () {
            _temporaryMessage(
              context,
              'Credentials screen will be connected next.',
            );
          },
        ),
        _divider(),
        _actionRow(
          icon: Icons.key_rounded,
          title: 'Reset Password',
          subtitle:
              'Generate a new temporary password',
          color: const Color(0xFF3478B7),
          onTap: () {
            _resetPassword(context);
          },
        ),
        _divider(),
        _actionRow(
          icon: Icons.history_rounded,
          title: 'Login Activity',
          subtitle:
              'View recent account access',
          color: const Color(0xFF7D55B4),
          onTap: () {
            _temporaryMessage(context, 'Use Owner Settings → Audit Log to review protected owner actions.');
          },
        ),
        _divider(),
        _actionRow(
          icon: isActive
              ? Icons.lock_outline_rounded
              : Icons.lock_open_rounded,
          title: isActive
              ? 'Disable Login'
              : 'Enable Login',
          subtitle: isActive
              ? 'Temporarily prevent student access'
              : 'Restore account access',
          color: const Color(0xFFC07421),
          onTap: () {
            _setAccountStatus(context, !isActive);
          },
        ),
      ],
    );
  }

  Widget _buildDangerZone(
    BuildContext context, {
    required bool isActive,
  }) {
    return Container(
      padding: const EdgeInsets.all(17),
      decoration: BoxDecoration(
        color: Colors.red.shade50,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: Colors.red.shade100,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.warning_amber_rounded,
                color: Colors.red.shade600,
                size: 20,
              ),
              const SizedBox(width: 8),
              Text(
                'Student Management',
                style: GoogleFonts.dmSans(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: Colors.red.shade700,
                ),
              ),
            ],
          ),

          const SizedBox(height: 7),

          Text(
            isActive
                ? 'Deactivate the student when they leave the hostel. Their historical records will remain available.'
                : 'This student account is currently inactive.',
            style: GoogleFonts.dmSans(
              fontSize: 10.5,
              height: 1.5,
              color: Colors.red.shade500,
            ),
          ),

          const SizedBox(height: 15),

          SizedBox(
            width: double.infinity,
            height: 46,
            child: OutlinedButton.icon(
              onPressed: () {
                _confirmDeactivate(
                  context,
                  isActive: isActive,
                );
              },
              icon: Icon(
                isActive
                    ? Icons.person_off_outlined
                    : Icons.person_add_alt_rounded,
                size: 17,
              ),
              label: Text(
                isActive
                    ? 'Deactivate Student'
                    : 'Reactivate Student',
                style: GoogleFonts.dmSans(
                  fontWeight: FontWeight.w700,
                  fontSize: 11.5,
                ),
              ),
              style: OutlinedButton.styleFrom(
                foregroundColor: Colors.red.shade600,
                side: BorderSide(
                  color: Colors.red.shade300,
                ),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _sectionCard({
    required List<Widget> children,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: 16,
        vertical: 4,
      ),
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: AppTheme.border,
          width: 0.6,
        ),
      ),
      child: Column(
        children: children,
      ),
    );
  }

  Widget _infoRow({
    required IconData icon,
    required String label,
    required String value,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(
        vertical: 13,
      ),
      child: Row(
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.06),
              borderRadius: BorderRadius.circular(11),
            ),
            child: Icon(
              icon,
              size: 18,
              color: AppTheme.teal,
            ),
          ),

          const SizedBox(width: 12),

          Expanded(
            child: Column(
              crossAxisAlignment:
                  CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: GoogleFonts.dmSans(
                    fontSize: 9.5,
                    color: AppTheme.textMuted,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  value,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600,
                    color: AppTheme.textPrimary,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _actionRow({
    required IconData icon,
    required String title,
    required String subtitle,
    required Color color,
    required VoidCallback onTap,
  }) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(14),
        child: Padding(
          padding: const EdgeInsets.symmetric(
            vertical: 12,
          ),
          child: Row(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: color.withOpacity(0.08),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(
                  icon,
                  size: 19,
                  color: color,
                ),
              ),

              const SizedBox(width: 12),

              Expanded(
                child: Column(
                  crossAxisAlignment:
                      CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: GoogleFonts.dmSans(
                        fontSize: 12.5,
                        fontWeight: FontWeight.w600,
                        color: AppTheme.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      subtitle,
                      style: GoogleFonts.dmSans(
                        fontSize: 9.5,
                        color: AppTheme.textMuted,
                      ),
                    ),
                  ],
                ),
              ),

              Icon(
                Icons.chevron_right_rounded,
                color: AppTheme.textMuted,
                size: 20,
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _divider() {
    return Divider(
      height: 1,
      thickness: 0.6,
      color: AppTheme.border,
    );
  }

  Future<void> _resetPassword(BuildContext context) async {
    final studentId = int.tryParse(student['id']?.toString() ?? '');
    if (studentId == null) { _temporaryMessage(context, 'Student record is missing its ID.'); return; }
    try {
      final resets = await Api().resetOwnerUserPasswords([studentId]);
      if (resets.isEmpty) throw Exception('Password reset was not permitted.');
      _temporaryMessage(context, 'Temporary password: ${resets.first['temporary_password']}');
    } catch (error) { _temporaryMessage(context, error.toString().replaceFirst('Exception: ', '')); }
  }

  Future<void> _setAccountStatus(BuildContext context, bool enabled) async {
    final studentId = int.tryParse(student['id']?.toString() ?? '');
    if (studentId == null) { _temporaryMessage(context, 'Student record is missing its ID.'); return; }
    try {
      await Api().setOwnerUserAccountStatus(studentId, enabled);
      student['status'] = enabled ? 'Active' : 'Disabled';
      _temporaryMessage(context, enabled ? 'Student login enabled.' : 'Student login disabled and active sessions revoked.');
      Navigator.of(context).maybePop();
    } catch (error) { _temporaryMessage(context, error.toString().replaceFirst('Exception: ', '')); }
  }

  Future<void> _showEditStudent(BuildContext context) async {
    final studentId = int.tryParse(student['id']?.toString() ?? '');
    if (studentId == null) {
      _temporaryMessage(context, 'Student record is missing its ID.');
      return;
    }
    final controllers = <String, TextEditingController>{
      'name': TextEditingController(text: student['name']?.toString() ?? ''),
      'email': TextEditingController(text: student['email']?.toString() ?? ''),
      'phone': TextEditingController(text: student['phone']?.toString() ?? ''),
      'cnic': TextEditingController(text: student['cnic']?.toString() ?? ''),
      'university': TextEditingController(text: student['university']?.toString() ?? ''),
      'department': TextEditingController(text: student['department']?.toString() ?? ''),
      'semester': TextEditingController(text: student['semester']?.toString() ?? ''),
      'parent_name': TextEditingController(text: student['guardian']?.toString() ?? ''),
      'parent_relation': TextEditingController(text: student['parentRelation']?.toString() ?? ''),
      'parent_phone': TextEditingController(text: student['parentPhone']?.toString() ?? ''),
      'parent_email': TextEditingController(text: student['parentEmail']?.toString() ?? ''),
      'parent_city': TextEditingController(text: student['city']?.toString() ?? ''),
    };
    var saving = false;
    await showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: AppTheme.bg,
      builder: (sheetContext) => StatefulBuilder(
        builder: (sheetContext, setSheetState) => SafeArea(
          child: Padding(
            padding: EdgeInsets.fromLTRB(
              20,
              14,
              20,
              20 + MediaQuery.viewInsetsOf(sheetContext).bottom,
            ),
            child: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Center(child: Container(width: 42, height: 4, decoration: BoxDecoration(color: AppTheme.border, borderRadius: BorderRadius.circular(4)))),
                  const SizedBox(height: 18),
                  Text('Edit student & parent', style: GoogleFonts.playfairDisplay(fontSize: 23, fontWeight: FontWeight.w700, color: AppTheme.textPrimary)),
                  const SizedBox(height: 4),
                  Text('Update the saved student and guardian information.', style: GoogleFonts.dmSans(fontSize: 11, color: AppTheme.textMuted)),
                  const SizedBox(height: 18),
                  ...[
                    ['name', 'Student name'], ['email', 'Student email'], ['phone', 'Student phone'], ['cnic', 'CNIC / B-Form'],
                    ['university', 'University'], ['department', 'Department'], ['semester', 'Semester'],
                    ['parent_name', 'Parent / guardian name'], ['parent_relation', 'Relationship'], ['parent_phone', 'Parent / guardian phone'], ['parent_email', 'Parent / guardian email'], ['parent_city', 'Parent / guardian city'],
                  ].map((field) => Padding(
                    padding: const EdgeInsets.only(bottom: 10),
                    child: TextField(
                      controller: controllers[field[0]],
                      keyboardType: field[0].contains('email') ? TextInputType.emailAddress : field[0].contains('phone') ? TextInputType.phone : TextInputType.text,
                      decoration: InputDecoration(labelText: field[1], border: OutlineInputBorder(borderRadius: BorderRadius.circular(14))),
                    ),
                  )),
                  const SizedBox(height: 8),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed: saving ? null : () async {
                        if (controllers['name']!.text.trim().isEmpty || controllers['email']!.text.trim().isEmpty) {
                          _temporaryMessage(context, 'Student name and email are required.');
                          return;
                        }
                        setSheetState(() => saving = true);
                        try {
                          await Api().updateOwnerStudent(
                            studentId: studentId,
                            updates: {for (final entry in controllers.entries) entry.key: entry.value.text.trim()},
                          );
                          student.addAll({
                            'name': controllers['name']!.text.trim(), 'email': controllers['email']!.text.trim(), 'phone': controllers['phone']!.text.trim(),
                            'cnic': controllers['cnic']!.text.trim(), 'university': controllers['university']!.text.trim(), 'department': controllers['department']!.text.trim(), 'semester': controllers['semester']!.text.trim(),
                            'guardian': controllers['parent_name']!.text.trim(), 'parentRelation': controllers['parent_relation']!.text.trim(), 'parentPhone': controllers['parent_phone']!.text.trim(), 'parentEmail': controllers['parent_email']!.text.trim(), 'city': controllers['parent_city']!.text.trim(),
                          });
                          if (sheetContext.mounted) Navigator.pop(sheetContext);
                          _temporaryMessage(context, 'Student and parent details saved.');
                        } catch (error) {
                          if (sheetContext.mounted) setSheetState(() => saving = false);
                          _temporaryMessage(context, error.toString().replaceFirst('Exception: ', ''));
                        }
                      },
                      style: ElevatedButton.styleFrom(backgroundColor: AppTheme.teal, foregroundColor: Colors.white, padding: const EdgeInsets.symmetric(vertical: 14)),
                      icon: saving ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white)) : const Icon(Icons.save_outlined),
                      label: Text(saving ? 'Saving...' : 'Save details'),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
    for (final controller in controllers.values) { controller.dispose(); }
  }

  void _showMoreMenu(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (_) {
        return SafeArea(
          top: false,
          child: Container(
            margin: const EdgeInsets.all(12),
            padding: const EdgeInsets.fromLTRB(
              18,
              12,
              18,
              18,
            ),
            decoration: BoxDecoration(
              color: AppTheme.bg,
              borderRadius: BorderRadius.circular(26),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 42,
                  height: 4,
                  margin:
                      const EdgeInsets.only(bottom: 14),
                  decoration: BoxDecoration(
                    color: AppTheme.border,
                    borderRadius: BorderRadius.circular(3),
                  ),
                ),

                _menuTile(
                  context,
                  icon: Icons.edit_outlined,
                  title: 'Edit Student',
                  onTap: () {
                    Navigator.pop(context);
                    _showEditStudent(context);
                  },
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _menuTile(
    BuildContext context, {
    required IconData icon,
    required String title,
    required VoidCallback onTap,
  }) {
    return ListTile(
      onTap: onTap,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(14),
      ),
      leading: Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          color: AppTheme.teal.withOpacity(0.07),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Icon(
          icon,
          color: AppTheme.teal,
          size: 19,
        ),
      ),
      title: Text(
        title,
        style: GoogleFonts.dmSans(
          fontSize: 12.5,
          fontWeight: FontWeight.w600,
          color: AppTheme.textPrimary,
        ),
      ),
      trailing: const Icon(
        Icons.chevron_right_rounded,
      ),
    );
  }

  void _confirmDeactivate(
    BuildContext context, {
    required bool isActive,
  }) {
    showDialog(
      context: context,
      builder: (dialogContext) {
        return AlertDialog(
          backgroundColor: AppTheme.bg,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(22),
          ),
          title: Text(
            isActive
                ? 'Deactivate Student?'
                : 'Reactivate Student?',
            style: GoogleFonts.playfairDisplay(
              fontSize: 20,
              fontWeight: FontWeight.w700,
              color: AppTheme.textPrimary,
            ),
          ),
          content: Text(
            isActive
                ? 'The student will no longer be able to login, but their records will remain available.'
                : 'This student will regain access to their StayBuddy account.',
            style: GoogleFonts.dmSans(
              fontSize: 12,
              height: 1.5,
              color: AppTheme.textMuted,
            ),
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(dialogContext);
              },
              child: Text(
                'Cancel',
                style: GoogleFonts.dmSans(
                  color: AppTheme.textMuted,
                ),
              ),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.pop(dialogContext);

                _temporaryMessage(
                  context,
                  isActive
                      ? 'Student deactivation will be connected to backend later.'
                      : 'Student reactivation will be connected to backend later.',
                );
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: isActive
                    ? Colors.red.shade600
                    : AppTheme.teal,
                foregroundColor: Colors.white,
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(13),
                ),
              ),
              child: Text(
                isActive
                    ? 'Deactivate'
                    : 'Reactivate',
                style: GoogleFonts.dmSans(
                  fontWeight: FontWeight.w700,
                  fontSize: 11.5,
                ),
              ),
            ),
          ],
        );
      },
    );
  }

  void _temporaryMessage(
    BuildContext context,
    String message,
  ) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          behavior: SnackBarBehavior.floating,
          backgroundColor: const Color(0xFF123F40),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
          content: Text(
            message,
            style: GoogleFonts.dmSans(
              color: Colors.white,
            ),
          ),
        ),
      );
  }
}
